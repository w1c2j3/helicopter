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
        "--models",
        model,
        "--tasks",
        task,
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
) -> int:
    """Run one model's queue sequentially and persist progress per model."""

    model_state_path = state_dir / f"{slug(model)}.json"
    model_state: dict[str, Any] = {"model": model, "jobs": []}
    if model_state_path.is_file():
        try:
            model_state = json.loads(model_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    failures = 0
    # A failed child must remain resumable.  Older state files incorrectly
    # advanced last_completed_index for every return code, so derive the
    # checkpoint from successful job records first.
    successful_indices = [
        int(item.get("index", -1))
        for item in model_state.get("jobs", [])
        if int(item.get("return_code", 1)) == 0
    ]
    if successful_indices:
        completed_index = max(successful_indices)
    elif not model_state.get("jobs"):
        completed_index = int(model_state.get("last_completed_index", -1))
    else:
        completed_index = -1
    effective_start = max(start_at, completed_index + 1)
    for job_index, (entry, mode) in enumerate(jobs):
        if job_index < effective_start:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    parser.add_argument("--proxy", help="HTTP proxy for dataset/judge downloads")
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--results-dir",
        default="results/large_eval_60",
        help="directory for logs, reports, and launcher state",
    )
    args = parser.parse_args()

    import tomllib

    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    run = manifest["run"]
    benchmarks = list(manifest["benchmarks"])
    if len(benchmarks) != 60:
        raise SystemExit(f"expected 60 benchmarks, found {len(benchmarks)}")
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
    models = [str(item) for item in run["models"]]
    tunnels = ensure_tunnels(endpoints, api_key)
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

        base_modes = [str(item) for item in run.get("base_modes", ["normal_nocot"])]
        cot_fields = {str(item) for item in run.get("cot_fields", [])}
        configured_cot_modes = run.get("cot_modes")
        if configured_cot_modes is None:
            configured_cot_modes = [run.get("cot_mode", "normal_cot")]
        cot_modes = [str(item) for item in configured_cot_modes]
        model_catalog = str(run["model_catalog"])
        jobs = build_job_queue(
            benchmarks,
            base_modes=base_modes,
            cot_fields=cot_fields,
            cot_modes=cot_modes,
        )
        print(
            f"large-eval: starting independent model queues; jobs_per_model={len(jobs)}",
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
