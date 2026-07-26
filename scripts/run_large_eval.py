"""Run the curated 60-benchmark suite against the four remote endpoints.

Every invocation is database-first: the child CLI is run with scoreboard
enabled, and the remote .env is loaded only into the process environment.
Each model owns an independent sequential queue, so a model moves to its next
benchmark/mode as soon as its current one finishes instead of waiting for the
other models.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/large_eval_60.toml"


def resolve_env_file(root: Path, requested: str | None) -> Path:
    if requested:
        path = root / requested
        if not path.is_file():
            raise SystemExit(f"required environment file not found: {path}")
        return path
    for name in (".env.remote", ".env.local", ".env"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no runtime environment file found under {root} (.env.remote/.env.local/.env)")


def load_dotenv(path: Path, env: dict[str, str]) -> None:
    if not path.is_file():
        raise SystemExit(f"required environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env.setdefault(key, value)


def configure_proxy(env: dict[str, str], proxy_url: str) -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = proxy_url
    no_proxy = [item.strip() for item in env.get("NO_PROXY", "").split(",") if item.strip()]
    for item in ("127.0.0.1", "localhost"):
        if item not in no_proxy:
            no_proxy.append(item)
    env["NO_PROXY"] = ",".join(no_proxy)
    env["no_proxy"] = env["NO_PROXY"]


def now() -> str:
    return datetime.now(UTC).isoformat()


def slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def check_endpoint(url: str, api_key: str | None) -> None:
    request = urllib.request.Request(f"{url.rstrip('/')}/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=10) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"endpoint {url} returned HTTP {response.status}")


def ensure_tunnels(endpoints: dict[str, str], api_key: str | None) -> list[subprocess.Popen[bytes]]:
    """Keep only the four requested local forwards alive for this run."""

    created: list[subprocess.Popen[bytes]] = []
    for url in endpoints.values():
        try:
            check_endpoint(url, api_key)
            continue
        except Exception:
            pass
        parsed = urlparse(url)
        if parsed.port is None:
            raise RuntimeError(f"endpoint has no port: {url}")
        port = int(parsed.port)
        process = subprocess.Popen(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-N",
                "-L",
                f"127.0.0.1:{port}:127.0.0.1:{port}",
                "rwkv-157",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        created.append(process)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"SSH forward for port {port} exited with {process.returncode}")
            try:
                check_endpoint(url, api_key)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"SSH forward for port {port} did not become healthy")
    return created


def close_tunnels(tunnels: list[subprocess.Popen[bytes]]) -> None:
    for process in tunnels:
        if process.poll() is None:
            process.terminate()
    for process in tunnels:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


async def check_database(root: Path) -> None:
    sys.path.insert(0, str(root / "src/scoreboard-server"))
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    await close_db()


def load_scored_job_keys(
    *,
    env: dict[str, str],
    root: Path,
    model_catalog: str,
    model: str,
) -> set[tuple[str, str]]:
    """Return scored (config filename, prompt mode) jobs for one model.

    A successful child process is not sufficient evidence that a benchmark is
    finished: generation may have been interrupted before eval rows or the
    aggregate score were written.  Only a task with a score and one eval row
    for every completion is eligible for skipping.
    """

    import tomllib

    catalog_path = Path(model_catalog)
    if not catalog_path.is_absolute():
        catalog_path = root / catalog_path
    catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    served_model_name = str(catalog["models"][model]["served_model_name"])

    required = (
        "SCOREBOARD_DB_HOST",
        "SCOREBOARD_DB_PORT",
        "SCOREBOARD_DB_USER",
        "SCOREBOARD_DB_PASSWORD",
        "SCOREBOARD_DB_NAME",
    )
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError("database skip check missing environment keys: " + ", ".join(missing))

    async def query() -> list[tuple[str, str]]:
        import asyncpg

        conn = await asyncpg.connect(
            host=env["SCOREBOARD_DB_HOST"],
            port=int(env["SCOREBOARD_DB_PORT"]),
            user=env["SCOREBOARD_DB_USER"],
            password=env["SCOREBOARD_DB_PASSWORD"],
            database=env["SCOREBOARD_DB_NAME"],
        )
        try:
            rows = await conn.fetch(
                """
                SELECT t.config_path, t.sampling_config
                FROM task t
                JOIN model m ON m.model_id = t.model_id
                JOIN scores s ON s.task_id = t.task_id
                JOIN completions c ON c.task_id = t.task_id
                LEFT JOIN eval e ON e.completions_id = c.completions_id
                WHERE t.status = 'Completed'
                  AND m.model_name = $1
                GROUP BY t.task_id, t.config_path, t.sampling_config
                HAVING count(c.completions_id) > 0
                   AND count(e.eval_id) = count(c.completions_id)
                """,
                served_model_name,
            )
        finally:
            await conn.close()

        keys: list[tuple[str, str]] = []
        for row in rows:
            sampling = row["sampling_config"]
            if isinstance(sampling, str):
                sampling = json.loads(sampling)
            mode = str((sampling or {}).get("prompt_mode", ""))
            if mode:
                keys.append((Path(str(row["config_path"])).name, mode))
        return keys

    return set(asyncio.run(query()))


def run_one(
    *,
    root: Path,
    env: dict[str, str],
    entry: dict[str, Any],
    mode: str,
    model: str,
    model_catalog: str,
    report_dir: Path,
    log_dir: Path,
    max_retries: int,
) -> int:
    task = str(entry["task"])
    config = str(entry["config"])
    label = f"{model}/{entry['field']}/{task}/{mode}"
    stem = slug(f"{model}_{entry['field']}_{task}_{mode}")
    report = report_dir / f"{stem}.json"
    log_path = log_dir / f"{stem}.log"
    command = [
        sys.executable,
        "-m",
        "helicopter_cli",
        "eval",
        "batch",
        "--config",
        config,
        "--model-catalog",
        model_catalog,
        "--job",
        f"{model}={task}",
        "--prompt-mode",
        mode,
            "--no-server",
            "--scoreboard",
            "--rerun",
        "--max-retries",
        str(max_retries),
        "--parallel",
        "1",
        "--batch-output",
        str(report),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"large-eval: START {label}", flush=True)
    print(f"large-eval: database required; log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"started_at={now()}\n")
        log_file.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(f"[{label}] {line}", end="", flush=True)
        return_code = process.wait()
        log_file.write(f"ended_at={now()}\nreturn_code={return_code}\n")
    elapsed = time.monotonic() - started
    print(f"large-eval: END {label} rc={return_code} elapsed={elapsed:.1f}s", flush=True)
    return return_code


def build_job_queue(
    benchmarks: list[dict[str, Any]],
    *,
    base_modes: list[str],
    cot_fields: set[str],
    cot_modes: list[str],
) -> list[tuple[dict[str, Any], str]]:
    """Expand the manifest into the ordered queue owned by one model."""

    jobs: list[tuple[dict[str, Any], str]] = []
    for entry in benchmarks:
        if bool(entry.get("skip", False)):
            continue
        modes = list(base_modes)
        if str(entry["field"]) in cot_fields:
            for cot_mode in cot_modes:
                if cot_mode not in modes:
                    modes.append(cot_mode)
        jobs.extend((entry, mode) for mode in modes)
    return jobs


def run_model_queue(
    *,
    root: Path,
    env: dict[str, str],
    model: str,
    jobs: list[tuple[dict[str, Any], str]],
    model_catalog: str,
    report_dir: Path,
    log_dir: Path,
    state_dir: Path,
    start_at: int,
    max_retries: int,
    run_policy: str,
) -> int:
    """Run one model's queue sequentially and persist progress per model."""

    model_state_path = state_dir / f"{slug(model)}.json"
    model_state: dict[str, Any] = {"model": model, "jobs": []}
    if model_state_path.is_file():
        try:
            model_state = json.loads(model_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    if run_policy == "resume":
        db_scored_jobs = load_scored_job_keys(
            env=env,
            root=root,
            model_catalog=model_catalog,
            model=model,
        )
    else:
        db_scored_jobs = set()
        successful_jobs = set()
    print(
        f"large-eval: policy={run_policy}; DB-complete jobs to skip "
        f"model={model} count={len(db_scored_jobs)}",
        flush=True,
    )

    failures = 0
    # Resume by semantic job identity instead of queue index.  The suite can
    # intentionally change its enabled modes (for example, dropping the
    # normal wave after it has started); index-only checkpoints would then
    # either repeat completed naive jobs or skip the wrong jobs.
    successful_jobs = {
        (str(item.get("field", "")), str(item.get("task", "")), str(item.get("mode", "")))
        for item in model_state.get("jobs", [])
        if int(item.get("return_code", 1)) == 0
    }
    for job_index, (entry, mode) in enumerate(jobs):
        job_key = (str(entry["field"]), str(entry["task"]), str(mode))
        db_job_key = (Path(str(entry["config"])).name, str(mode))
        if job_index < int(start_at) or db_job_key in db_scored_jobs:
            if db_job_key in db_scored_jobs and job_key not in successful_jobs:
                print(
                    f"large-eval: SKIP DB-complete {model}/{entry['task']}/{mode}",
                    flush=True,
                )
            continue
        return_code = run_one(
            root=root,
            env=env,
            entry=entry,
            mode=mode,
            model=model,
            model_catalog=model_catalog,
            report_dir=report_dir,
            log_dir=log_dir,
            max_retries=max_retries,
        )
        model_state.setdefault("jobs", []).append(
            {
                "index": job_index,
                "field": entry["field"],
                "task": entry["task"],
                "mode": mode,
                "return_code": return_code,
                "ended_at": now(),
            }
        )
        if return_code == 0:
            model_state["last_completed_index"] = job_index
            model_state.pop("last_failed_index", None)
            if run_policy == "resume":
                successful_jobs.add(job_key)
        else:
            model_state["last_failed_index"] = job_index
        model_state_path.write_text(
            json.dumps(model_state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        failures += int(return_code != 0)
        if return_code != 0:
            print(
                f"large-eval: STOP {model} after failed job index={job_index}; "
                "the same job will be retried on the next launcher run",
                flush=True,
            )
            break
    print(f"large-eval: MODEL END {model} failures={failures}", flush=True)
    return failures


def select_mode_waves(
    run: dict[str, Any], *, use_naive: bool, use_normal: bool
) -> tuple[list[str], list[str]]:
    """Resolve mode waves from CLI flags without changing the suite manifest.

    No flag means naive-only.  When both flags are present, normal is placed
    first so it has priority while both waves remain in the same stable queue.
    """

    families: list[str] = []
    if not use_naive and not use_normal:
        use_naive = True
    if use_naive:
        families.append("naive")
    if use_normal:
        families.append("normal")

    configured_base = {str(item) for item in run.get("base_modes", [])}
    configured_cot = {str(item) for item in run.get("cot_modes", [])}
    base_modes = [f"{family}_nocot" for family in families if f"{family}_nocot" in configured_base]
    cot_modes = [f"{family}_cot" for family in families if f"{family}_cot" in configured_cot]
    if not base_modes:
        raise SystemExit("no requested NoCoT modes are available in the suite manifest")
    if not cot_modes and any(str(item) in {"math", "knowledge"} for item in run.get("cot_fields", [])):
        raise SystemExit("no requested CoT modes are available in the suite manifest")
    return base_modes, cot_modes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    parser.add_argument("--proxy", help="HTTP proxy for dataset/judge downloads")
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument(
        "--naive",
        action="store_true",
        help="include naive modes; default when no mode flag is supplied",
    )
    parser.add_argument(
        "--normal",
        action="store_true",
        help="include normal modes; they run after naive when combined with --naive",
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--resume",
        action="store_true",
        help="skip complete scored jobs and continue incomplete work (default)",
    )
    policy.add_argument(
        "--rerun",
        action="store_true",
        help="run selected jobs again even when history already has a score",
    )
    parser.add_argument(
        "--fields",
        help="comma-separated benchmark fields to include (default: all)",
    )
    parser.add_argument(
        "--tasks",
        help="comma-separated manifest task names to include (default: all)",
    )
    parser.add_argument(
        "--benchmark-start",
        type=int,
        default=None,
        help="inclusive manifest benchmark index (default: 0)",
    )
    parser.add_argument(
        "--benchmark-end",
        type=int,
        default=None,
        help="exclusive manifest benchmark index (default: end)",
    )
    parser.add_argument(
        "--models",
        help="comma-separated model aliases to run; defaults to all models in the manifest",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        default="results/large_eval_60",
        help="directory for logs, reports, and launcher state",
    )
    args = parser.parse_args()
    run_policy = "rerun" if args.rerun else "resume"

    import tomllib

    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    run = manifest["run"]
    benchmarks = list(manifest["benchmarks"])
    if len(benchmarks) != 60:
        raise SystemExit(f"expected 60 benchmarks, found {len(benchmarks)}")
    benchmark_start = max(0, int(args.benchmark_start or 0))
    benchmark_end = args.benchmark_end
    if benchmark_end is not None and benchmark_end < benchmark_start:
        raise SystemExit("--benchmark-end must be greater than or equal to --benchmark-start")
    benchmarks = benchmarks[benchmark_start:benchmark_end]
    if args.fields:
        fields = {item.strip() for item in str(args.fields).split(",") if item.strip()}
        benchmarks = [item for item in benchmarks if str(item.get("field")) in fields]
    if args.tasks:
        tasks = {item.strip() for item in str(args.tasks).split(",") if item.strip()}
        benchmarks = [item for item in benchmarks if str(item.get("task")) in tasks]
    if not benchmarks:
        raise SystemExit("benchmark filters selected no manifest entries")
    env = dict(os.environ)
    env_file = resolve_env_file(ROOT, args.env_file)
    load_dotenv(env_file, env)
    print(f"large-eval: loaded runtime environment from {env_file.name}", flush=True)
    configure_proxy(env, str(args.proxy or run["proxy_url"]))
    env["HELICOPTER_REQUIRE_SCOREBOARD"] = "1"
    env["HELICOPTER_SCOREBOARD_DB_ONLY"] = "1"
    request_concurrency = int(run.get("request_concurrency", 16))
    if request_concurrency <= 0:
        raise SystemExit("[run].request_concurrency must be a positive integer")
    # The child batch command already resolves this override alongside the
    # benchmark TOML. Keeping it in the suite manifest makes the throughput
    # policy explicit without changing sampling or scoreboard identity.
    env["HELICOPTER_EVAL_CONCURRENT_REQUESTS"] = str(request_concurrency)

    api_key = env.get("HELICOPTER_EVAL_API_KEY") or env.get("OPENAI_API_KEY")
    endpoints = {
        "g1h-1.5b": "http://127.0.0.1:19315/v1",
        "g1h-2.9b": "http://127.0.0.1:19329/v1",
        "g1h-7.2b": "http://127.0.0.1:29572/v1",
        "g1h-13.3b": "http://127.0.0.1:29533/v1",
    }
    configured_models = [str(item) for item in run["models"]]
    if args.models:
        models = [item.strip() for item in str(args.models).split(",") if item.strip()]
        unknown_models = sorted(set(models) - set(configured_models))
        if unknown_models:
            raise SystemExit(
                "unknown model aliases in --models: "
                + ", ".join(unknown_models)
                + "; configured aliases: "
                + ", ".join(configured_models)
            )
        if not models:
            raise SystemExit("--models must contain at least one model alias")
    else:
        models = configured_models
    selected_endpoints = {model: endpoints[model] for model in models}
    tunnels = ensure_tunnels(selected_endpoints, api_key)
    try:
        asyncio.run(check_database(ROOT))
        print("large-eval: database and all four model endpoints are healthy", flush=True)

        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = ROOT / results_dir
        report_dir = results_dir / "batch_reports"
        log_dir = results_dir / "logs"
        state_dir = results_dir / "state"
        report_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        cot_fields = {str(item) for item in run.get("cot_fields", [])}
        base_modes, cot_modes = select_mode_waves(
            run,
            use_naive=bool(args.naive),
            use_normal=bool(args.normal),
        )
        model_catalog = str(run["model_catalog"])
        jobs = build_job_queue(
            benchmarks,
            base_modes=base_modes,
            cot_fields=cot_fields,
            cot_modes=cot_modes,
        )
        print(
            "large-eval: selected modes "
            f"base={base_modes} cot={cot_modes}; "
            f"starting independent model queues; jobs_per_model={len(jobs)}",
            flush=True,
        )
        failures = 0
        with ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="model-queue") as executor:
            futures = [
                executor.submit(
                    run_model_queue,
                    root=ROOT,
                    env=env.copy(),
                    model=model,
                    jobs=jobs,
                    model_catalog=model_catalog,
                    report_dir=report_dir,
                    log_dir=log_dir,
                    state_dir=state_dir,
                    start_at=args.start_at,
                    max_retries=max(0, int(args.max_retries)),
                    run_policy=run_policy,
                )
                for model in models
            ]
            for future in as_completed(futures):
                failures += future.result()

        print(f"large-eval: completed model queues={len(models)} failures={failures}", flush=True)
        return 1 if failures else 0
    finally:
        close_tunnels(tunnels)


if __name__ == "__main__":
    raise SystemExit(main())
