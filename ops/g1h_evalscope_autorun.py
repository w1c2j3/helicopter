#!/usr/bin/env python3
"""Unattended EvalScope Agent runner for the four formal g1h models on 157."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path("/home/rwkv/chase/helicopter-e0eeddc")
PYTHON = ROOT / ".venv/bin/python"
EVALSCOPE = ROOT / ".venv/bin/evalscope"
TAU2_EVALSCOPE = Path("/home/rwkv/chase/EvalScope/.venv/bin/evalscope")
CONFIG = ROOT / "configs/models/g1h-dual-replica.toml"
SWE_CONFIG = ROOT / "configs/evalscope_agent/swebench_verified_mini_docker.toml"
TERMINAL_CONFIG = ROOT / "configs/evalscope_agent/terminal_bench_v2_1_docker.toml"
STATUS = Path("/home/rwkv/chase/eval-results/g1h_agent_status.py")
RUNTIME = Path("/home/rwkv/chase/eval-results/g1h-agent-autorun")
PRIVATE_CONFIG_DIR = Path(tempfile.gettempdir()) / "helicopter-evalscope"
REMOTE_ENV = Path("/home/rwkv/chase/helicopter/.env.remote")
AGENT_CONFIG = Path("/home/rwkv/chase/eval-results/.fc9_agent_config.json")
GENERATION_CONFIG = Path("/home/rwkv/chase/eval-results/.fc9_generation_config.json")
MODEL_ARGS = Path("/home/rwkv/chase/eval-results/.fc9_model_args.json")
OFFLINE_SWEBENCH = Path("/home/rwkv/chase/eval-results/swebench-offline-pythonpath")
TAU2_DATA = Path(
    "/home/rwkv/.cache/modelscope/hub/datasets/datasets/"
    "evalscope--tau2-bench-data/snapshots/master"
)
# ``batch`` is total EvalScope sample concurrency before the four-lane split.
# Parallel candidate routing can fan one sample into up to 12 model requests.
MODELS = {
    "1.5B": {
        "cli": "g1h-1.5b",
        "full": "rwkv7-g1h-1.5b-20260710-ctx10240",
        "port": 19415,
        "batch": 128,
    },
    "2.9B": {
        "cli": "g1h-2.9b",
        "full": "rwkv7-g1h-2.9b-20260710-ctx10240",
        "port": 19429,
        "batch": 96,
    },
    "7.2B": {
        "cli": "g1h-7.2b",
        "full": "rwkv7-g1h-7.2b-20260710-ctx10240",
        "port": 29572,
        "batch": 64,
    },
    "13.3B": {
        "cli": "g1h-13.3b",
        "full": "rwkv7-g1h-13.3b-20260710-ctx10240",
        "port": 29533,
        "batch": 48,
    },
}
AGGREGATE_ONLY_SAMPLES = {
    "k2_verifier": 2000,
    "minimax_verifier": 102,
}
GENERAL = {
    "browsecomp", "claw_eval", "gaia", "gdpval", "officeqa",
    "researchrubrics", "wide_search",
}
JUDGE_BACKED = GENERAL | {"mcp_atlas"}
DIRECT_RESPONSE = {"browsecomp", "officeqa"}
NEWAPI_HOST = "next-token.cc"
NEWAPI_BASE_URL = f"https://{NEWAPI_HOST}/v1"
SWE = {
    "deep_swe", "swe_bench_lite_agentic", "swe_bench_multilingual_agentic",
    "swe_bench_pro", "swe_bench_verified_agentic",
    "swe_bench_verified_mini_agentic",
}
TERMINAL = {"terminal_bench_v2", "terminal_bench_v2_1"}
TAU = {"tau2_bench", "tau_bench", "tau3_bench"}

# Fast/no-container datasets first. Environment-heavy datasets remain independent
# so one missing dependency cannot suppress the rest of the matrix.
ORDER = [
    "bfcl_v3", "bfcl_v4", "general_fc", "k2_verifier", "kimi_verifier",
    "minimax_verifier", "mcp_atlas", "browsecomp", "gaia", "officeqa",
    "researchrubrics", "wide_search", "skillsbench", "tau2_bench", "tau_bench",
    "tau3_bench", "deep_swe", "swe_bench_lite_agentic",
    "swe_bench_multilingual_agentic", "swe_bench_pro",
    "swe_bench_verified_agentic", "swe_bench_verified_mini_agentic",
    "terminal_bench_v2", "terminal_bench_v2_1", "claw_eval", "gdpval",
    "toolathlon",
]

# Temporarily deferred by operator request. Keep the dataset in ORDER so the
# catalog and CLI remain complete, but unattended workers must not schedule it.
DEFERRED = {"browsecomp"}

ROUTER_ARGS = [
    "--parallel-candidate-router",
    "--candidate-chunk-tools", "2",
    "--candidate-batch-size", "16",
    "--candidate-context-chars", "6000",
    "--candidate-prompt-max-chars", "12288",
    "--candidate-max-tokens", "2048",
    "--aggregate-max-tokens", "2048",
    "--candidate-max-candidates", "12",
    "--long-doc-min-chars", "6000",
    "--long-doc-max-chars", "1000",
    "--long-doc-overlap-lines", "3",
    "--long-doc-max-evidence-chunks", "4",
    "--long-doc-max-evidence-chars", "6000",
]


def load_remote_env() -> None:
    """Source the server-owned env file without putting secrets in argv or logs."""

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -a; source "$1"; env -0',
            "g1h-env-loader",
            str(REMOTE_ENV),
        ],
        check=True,
        capture_output=True,
    )
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        os.environ[key.decode()] = value.decode()


def log_event(event: str, **fields: object) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    print(line, flush=True)
    details = " ".join(
        f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}"
        for key, value in fields.items()
    )
    with (RUNTIME / "events.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{record['time']} {event}{(' ' + details) if details else ''}\n")


def status() -> dict:
    completed = subprocess.run(
        [str(PYTHON), str(STATUS)], check=True, text=True, capture_output=True, env=os.environ
    )
    return json.loads(completed.stdout)


def score_is_usable(row: dict) -> bool:
    completions = int(row.get("completions") or 0)
    evals = int(row.get("evals") or 0)
    if completions > 0 and completions == evals:
        return True
    benchmark = row.get("benchmark")
    expected = AGGREGATE_ONLY_SAMPLES.get(benchmark)
    if expected is None or completions != expected:
        return False
    audit = row.get("context_audit")
    if not isinstance(audit, dict) or int(audit.get("samples") or 0) != completions:
        return False
    return all(int(audit.get(key) or 0) == 0 for key in (
        "missing_reviews",
        "invalid_reviews",
        "context_error_samples",
        "inference_error_samples",
    ))


def scored_pairs() -> set[tuple[str, str]]:
    return {
        (row["model"], row["benchmark"])
        for row in status()["latest_scores"]
        if score_is_usable(row)
    }


def eval_batch_for_dataset(dataset: str, lane_batch: int) -> int:
    """Cap external-judge fanout without reducing core model concurrency."""

    if dataset not in JUDGE_BACKED:
        return lane_batch
    try:
        judge_limit = int(os.environ.get("HELICOPTER_JUDGE_CONCURRENT_REQUESTS", "5"))
    except ValueError:
        judge_limit = 5
    return max(1, min(lane_batch, judge_limit))


def model_busy(full_name: str) -> bool:
    completed = subprocess.run(
        ["pgrep", "-af", f"evalscope eval --model {full_name}"],
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(completed.stdout.strip())


def toolathlon_busy() -> bool:
    completed = subprocess.run(
        ["pgrep", "-af", "evalscope.benchmarks.toolathlon.ws_client"],
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(completed.stdout.strip())


def endpoint_ready(port: int) -> bool:
    completed = subprocess.run(
        [
            "curl", "-fsS", "--max-time", "10",
            "-H", "Authorization: Bearer rwkv-skills",
            f"http://127.0.0.1:{port}/v1/models",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def secure_runtime_configs(dataset: str) -> tuple[Path | None, Path | None]:
    key = os.environ.get("HELICOPTER_EXTERNAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_path = None
    dataset_path = None
    if dataset in GENERAL or dataset == "mcp_atlas":
        if not key:
            raise RuntimeError("external judge API key is not configured")
        judge_path = PRIVATE_CONFIG_DIR / f"judge-model-args-{os.getpid()}-{dataset}.json"
        write_private_json(
            judge_path,
            {
                "model_id": "gpt-4o-mini",
                "api_url": NEWAPI_BASE_URL,
                "api_key": key,
                "generation_config": {"temperature": 0.01, "max_tokens": 2048},
            },
        )

    extra: dict[str, dict] = {}
    if dataset == "mcp_atlas":
        extra[dataset] = {
            "extra_params": {
                "mcp_server_url": "http://127.0.0.1:1984",
                "filter_enabled_servers": True,
            }
        }
    elif dataset == "skillsbench":
        extra[dataset] = {
            "extra_params": {
                "tasks_dir": "/home/rwkv/chase/rwkv-skills/data/skillsbench/tasks",
                "skill_mode": "no-skill",
            }
        }
    elif dataset in TAU:
        if not key:
            raise RuntimeError("external user-model API key is not configured")
        params: dict[str, object] = {
            "user_model": "gpt-4o-mini",
            "api_key": key,
            "api_base": NEWAPI_BASE_URL,
            "generation_config": {"temperature": 0.01, "max_tokens": 2048},
        }
        if dataset == "tau3_bench":
            params["retrieval_config"] = "bm25"
        extra[dataset] = {"extra_params": params}
    elif dataset in SWE - {"deep_swe"}:
        extra[dataset] = {
            "extra_params": {
                "build_docker_images": True,
                "pull_remote_images_if_available": False,
            }
        }
    if extra:
        dataset_path = PRIVATE_CONFIG_DIR / f"dataset-args-{os.getpid()}-{dataset}.json"
        write_private_json(dataset_path, extra)
    return judge_path, dataset_path


def command_for(
    model_label: str,
    dataset: str,
    work: Path,
    cache: Path,
    *,
    smoke: bool,
    eval_batch_size: int | None = None,
) -> tuple[list[str], dict[str, str], list[Path]]:
    model = MODELS[model_label]
    binary = TAU2_EVALSCOPE if dataset == "tau2_bench" else EVALSCOPE
    config = CONFIG
    judge_path, dataset_path = secure_runtime_configs(dataset)
    command = [
        str(PYTHON), "-m", "helicopter_cli", "eval", "evalscope",
        "--config", str(config), str(model["cli"]), dataset,
        "--binary", str(binary),
        "--base-url", f"http://127.0.0.1:{model['port']}/v1",
        "--api-key", "rwkv-skills",
        "--mode", "native",
        "--no-server",
        "--no-naive-chat-proxy",
        *ROUTER_ARGS,
        "--generation-config", str(GENERATION_CONFIG),
        "--model-args", str(MODEL_ARGS),
        "--eval-batch-size", str(eval_batch_size or model["batch"]),
        "--work-dir", str(work),
        "--no-timestamp",
        "--use-cache", str(cache),
        "--ignore-errors",
        "--judge-strategy", "auto",
    ]
    if dataset in SWE:
        command[command.index("--config") + 1] = str(SWE_CONFIG)
        command += [
            "--strategy", "swe_bench_toolcall",
            "--tools", "bash",
            "--agent-environment", "docker",
            "--max-steps", "250" if not smoke else "20",
        ]
    elif dataset in TERMINAL:
        command[command.index("--config") + 1] = str(TERMINAL_CONFIG)
        command.append("--no-agent-config")
    elif dataset in DIRECT_RESPONSE:
        command.append("--no-agent-config")
    else:
        command += ["--agent-config", str(AGENT_CONFIG)]
    if judge_path:
        command += ["--judge-model-args", str(judge_path)]
    if dataset_path:
        command += ["--dataset-args", str(dataset_path)]
    if smoke:
        command += ["--limit", "1"]
    else:
        command += ["--scoreboard", "--scoreboard-db-only"]

    run_env = os.environ.copy()
    # The former LAN proxy at 192.168.0.243:7890 is no longer reachable.
    # GitHub, ModelScope, and the judge endpoint are directly reachable from
    # 157, so inherited proxy variables must not poison dataset/tool setup.
    for variable in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        run_env.pop(variable, None)
    for variable in ("NO_PROXY", "no_proxy"):
        hosts = [item.strip() for item in run_env.get(variable, "").split(",") if item.strip()]
        if NEWAPI_HOST not in hosts:
            hosts.append(NEWAPI_HOST)
        run_env[variable] = ",".join(hosts)
    run_env["PYTHONPATH"] = ":".join(
        [
            str(OFFLINE_SWEBENCH) if dataset in SWE else "",
            str(ROOT / "src/cli"),
            str(ROOT / "src/scoreboard-server"),
        ]
    ).strip(":")
    run_env["HF_HOME"] = "/home/rwkv/.cache/huggingface"
    run_env["HF_HUB_CACHE"] = "/home/rwkv/.cache/huggingface/hub"
    online_dataset = dataset == "claw_eval"
    run_env["HF_HUB_OFFLINE"] = "0" if online_dataset else "1"
    run_env["TRANSFORMERS_OFFLINE"] = "0" if online_dataset else "1"
    run_env["MODELSCOPE_OFFLINE"] = "0" if online_dataset else "1"
    if dataset == "tau2_bench":
        run_env["TAU2_DATA_DIR"] = str(TAU2_DATA)
    return command, run_env, [path for path in (judge_path, dataset_path) if path is not None]


def run_pair(
    model_label: str,
    dataset: str,
    *,
    smoke: bool = False,
    eval_batch_size: int | None = None,
) -> int:
    model = MODELS[model_label]
    if not endpoint_ready(int(model["port"])):
        log_event("endpoint_unavailable", model=model_label, dataset=dataset)
        return 75
    if not smoke and (model_label, dataset) in scored_pairs():
        log_event("already_scored", model=model_label, dataset=dataset)
        return 0

    work = RUNTIME / ("smoke" if smoke else "work") / model_label / dataset
    # Database rows are the only durable resume state. EvalScope still needs a
    # short-lived work tree while generating and scoring a benchmark.
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    cache = work
    log_path = RUNTIME / "logs" / f"{model_label}-{dataset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_configs: list[Path] = []
    try:
        command, run_env, temporary_configs = command_for(
            model_label,
            dataset,
            work,
            cache,
            smoke=smoke,
            eval_batch_size=eval_batch_size,
        )
    except Exception as error:  # environment/setup error remains retryable
        log_event("environment_paused", model=model_label, dataset=dataset, error=str(error))
        return 78

    lock_stream = None
    try:
        if dataset == "toolathlon":
            lock_stream = (RUNTIME / "toolathlon.lock").open("a+")
            fcntl.flock(lock_stream, fcntl.LOCK_EX)
            while toolathlon_busy():
                log_event("toolathlon_wait", model=model_label)
                time.sleep(60)
        log_event("start", model=model_label, dataset=dataset, smoke=smoke, log=str(log_path))
        with log_path.open("ab", buffering=0) as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if smoke:
            log_event("smoke_exit", model=model_label, dataset=dataset, returncode=completed.returncode)
            return completed.returncode
        row = next(
            (
                item
                for item in status()["latest_scores"]
                if item["model"] == model_label and item["benchmark"] == dataset
            ),
            None,
        )
        if row is None:
            log_event(
                "no_score",
                model=model_label,
                dataset=dataset,
                returncode=completed.returncode,
                log=str(log_path),
            )
            return completed.returncode or 70
        log_event(
            "scored",
            model=model_label,
            dataset=dataset,
            score=row["score"],
            completions=row["completions"],
            evals=row["evals"],
            fully_validated=row["fully_validated"],
        )
        if not score_is_usable(row):
            log_event(
                "unvalidated_score",
                model=model_label,
                dataset=dataset,
                completions=row["completions"],
                evals=row["evals"],
            )
            return 75
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        log_event("work_cleaned", model=model_label, dataset=dataset, smoke=smoke)
        for path in temporary_configs:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if lock_stream is not None:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
            lock_stream.close()


def run_worker(model_label: str, forever: bool, lane: int, lanes: int) -> int:
    if lane < 0 or lane >= lanes:
        raise ValueError(f"lane must satisfy 0 <= lane < lanes, got {lane}/{lanes}")
    model = MODELS[model_label]
    lane_order = [
        dataset
        for index, dataset in enumerate(ORDER)
        if index % lanes == lane and dataset not in DEFERRED
    ]
    lane_batch = max(4, math.ceil(int(model["batch"]) / lanes))
    round_number = 0
    while True:
        round_number += 1
        current = scored_pairs()
        missing = [dataset for dataset in lane_order if (model_label, dataset) not in current]
        if not missing:
            log_event("lane_complete", model=model_label, lane=lane, lanes=lanes)
            return 0
        log_event(
            "round_start",
            model=model_label,
            lane=lane,
            lanes=lanes,
            batch=lane_batch,
            round=round_number,
            missing=len(missing),
        )
        progress = False
        retryable = False
        for dataset in missing:
            before = (model_label, dataset) in scored_pairs()
            result = run_pair(
                model_label,
                dataset,
                eval_batch_size=eval_batch_for_dataset(dataset, lane_batch),
            )
            after = (model_label, dataset) in scored_pairs()
            progress = progress or (not before and after)
            if result != 0:
                retryable = True
                time.sleep(300)
        if not forever:
            return 0 if progress else 1
        remaining = len(
            [
                pair
                for pair in status()["missing"]
                if pair["model"] == model_label and pair["benchmark"] in lane_order
            ]
        )
        log_event(
            "round_end",
            model=model_label,
            lane=lane,
            lanes=lanes,
            round=round_number,
            remaining=remaining,
        )
        if remaining == 0:
            return 0
        time.sleep(1800 if progress else (300 if retryable else 1800))


def process_present(pattern: str) -> bool:
    completed = subprocess.run(
        ["pgrep", "-af", pattern], check=False, text=True, capture_output=True
    )
    return bool(completed.stdout.strip())


def self_check() -> int:
    catalog = json.loads(
        (ROOT / "benchmarks/evalscope_agent_datasets.json").read_text(encoding="utf-8")
    )
    expected = {item["name"] for item in catalog["datasets"]}
    queued = set(ORDER)
    generation = json.loads(GENERATION_CONFIG.read_text(encoding="utf-8"))
    current_status = status()
    result = {
        "queue_coverage_ok": queued == expected,
        "queue_count": len(queued),
        "catalog_count": len(expected),
        "queue_missing": sorted(expected - queued),
        "queue_extra": sorted(queued - expected),
        "deferred": sorted(DEFERRED),
        "workers": {
            label: process_present(f"g1h_evalscope_autorun.py worker {label}")
            for label in MODELS
        },
        "endpoints": {
            label: endpoint_ready(int(model["port"]))
            for label, model in MODELS.items()
        },
        "formal_database_port": os.environ.get("SCOREBOARD_DB_PORT"),
        "temperature": generation.get("temperature"),
        "max_tokens": generation.get("max_tokens"),
        "scoreboard_db_only": True,
        "official_score_pairs": current_status["official_score_pairs"],
        "remaining_pairs": current_status["remaining_pairs"],
        "failure_policy": "continue_independently_then_retry",
        "toolathlon_policy": "global_serial_lock",
    }
    result["ok"] = (
        result["queue_coverage_ok"]
        and all(result["workers"].values())
        and all(result["endpoints"].values())
        and result["formal_database_port"] == "55433"
        and result["temperature"] == 0.01
        and result["scoreboard_db_only"]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("model", choices=MODELS)
    smoke_parser.add_argument("dataset", choices=ORDER)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("model", choices=MODELS)
    worker_parser.add_argument("--once", action="store_true")
    worker_parser.add_argument("--lane", type=int, default=0)
    worker_parser.add_argument("--lanes", type=int, default=1)
    subparsers.add_parser("check")
    return parser.parse_args()


def main() -> int:
    load_remote_env()
    args = parse_args()
    if args.mode == "smoke":
        return run_pair(args.model, args.dataset, smoke=True)
    if args.mode == "check":
        return self_check()
    return run_worker(
        args.model,
        forever=not args.once,
        lane=args.lane,
        lanes=args.lanes,
    )


if __name__ == "__main__":
    sys.exit(main())
