from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import requests

from .benchmark_catalog_defaults import (
    CATALOG_RUN_STATUS,
    CATALOG_SCOPE,
    CATALOG_SOURCE,
    CATALOG_TARGET_KIND,
)
from .commands import build_lighteval_plan, local_openai_base_url, resolve_model_entry, table
from .env import env_value, pick
from .eval_run import (
    SCOREBOARD_LOCK,
    _scoreboard_env,
    run_eval,
    scoreboard_dataset_name,
    scoreboard_model_name,
)
from .function_calling import run_function_calling_eval


DEFAULT_GPU_IDLE_MIB = 2048.0
DEFAULT_PORT_BASE = 8000
OPEN_FILE_RESERVE_MIN = 128
OPEN_FILES_PER_REQUEST = 4

UNIT_KINDS = ("lighteval", "fc")


@dataclass
class BatchUnit:
    model: str
    kind: str  # one of UNIT_KINDS
    tasks: list[str]
    status: str = "pending"
    message: str = ""
    attempts: int = 0
    elapsed_seconds: float = 0.0
    skipped_tasks: list[str] = field(default_factory=list)
    slot_index: int | None = None
    gpu: int | None = None
    port: int | None = None
    exit_code: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    replica: str | None = None


@dataclass
class GpuSlot:
    index: int
    gpu: int | None
    port: int


@dataclass(frozen=True)
class ModelReplica:
    name: str
    base_url: str | None = None
    served_model_name: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    max_num_seqs: int | None = None


@dataclass(frozen=True)
class ModelConcurrency:
    model: str
    benchmark_workers: int
    concurrent_requests: int
    rollout_n: int
    max_num_seqs: int | None
    running_requests: int
    waiting_requests: int
    source: str


def run_model_aware_scheduler(
    units: list[BatchUnit],
    *,
    model_worker_limits: dict[str, int],
    max_workers: int,
    worker: Callable[[BatchUnit], None],
) -> None:
    """Run generation without letting waits for one model starve another.

    A global executor that submits every unit eagerly can spend all of its
    threads waiting on replicas of one busy model. Keep model ownership in the
    dispatcher instead, and submit a unit only while that model has capacity.
    """

    if not units:
        return
    max_workers = max(1, min(int(max_workers), len(units)))
    model_order = list(dict.fromkeys(unit.model for unit in units))
    pending: dict[str, deque[BatchUnit]] = {
        model: deque(unit for unit in units if unit.model == model)
        for model in model_order
    }
    limits = {
        model: max(1, int(model_worker_limits.get(model, 1)))
        for model in model_order
    }
    active_by_model = {model: 0 for model in model_order}
    active: dict[Future[None], str] = {}
    cursor = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while active or any(pending.values()):
            while len(active) < max_workers and any(pending.values()):
                selected_model: str | None = None
                for offset in range(len(model_order)):
                    position = (cursor + offset) % len(model_order)
                    model = model_order[position]
                    if pending[model] and active_by_model[model] < limits[model]:
                        selected_model = model
                        cursor = (position + 1) % len(model_order)
                        break
                if selected_model is None:
                    break
                unit = pending[selected_model].popleft()
                future = executor.submit(worker, unit)
                active[future] = selected_model
                active_by_model[selected_model] += 1

            if not active:
                remaining = {
                    model: len(queue_)
                    for model, queue_ in pending.items()
                    if queue_
                }
                raise RuntimeError(
                    f"model-aware scheduler cannot dispatch pending units: {remaining}"
                )

            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            completed_futures: list[Future[None]] = []
            for future in completed:
                model = active.pop(future)
                active_by_model[model] -= 1
                completed_futures.append(future)
            for future in completed_futures:
                future.result()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_as_str_list(item))
        return result
    return [str(value)]


def resolve_model_replicas(config: dict[str, Any], model: str) -> list[ModelReplica]:
    """Return independently schedulable endpoint replicas for one logical model."""

    entry = resolve_model_entry(config, model)
    raw_replicas = entry.get("replicas")
    if raw_replicas is None:
        return [
            ModelReplica(
                name="default",
                base_url=str(entry["base_url"]) if entry.get("base_url") else None,
                served_model_name=str(entry["served_model_name"]) if entry.get("served_model_name") else None,
                api_key=str(entry["api_key"]) if entry.get("api_key") else None,
                api_key_env=str(entry["api_key_env"]) if entry.get("api_key_env") else None,
                max_num_seqs=int(entry["max_num_seqs"]) if entry.get("max_num_seqs") else None,
            )
        ]
    if not isinstance(raw_replicas, list) or not raw_replicas:
        raise SystemExit(f"model {model} replicas must be a non-empty TOML array of tables")

    replicas: list[ModelReplica] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_replicas, start=1):
        if not isinstance(raw, dict):
            raise SystemExit(f"model {model} replica {index} must be a TOML table")
        name = str(raw.get("name") or f"replica-{index}").strip()
        if not name or name in seen_names:
            raise SystemExit(f"model {model} replica names must be non-empty and unique")
        seen_names.add(name)
        base_url = str(raw.get("base_url") or "").strip()
        if not base_url:
            raise SystemExit(f"model {model} replica {name!r} requires base_url")
        max_num_seqs = raw.get("max_num_seqs", entry.get("max_num_seqs"))
        replicas.append(
            ModelReplica(
                name=name,
                base_url=base_url,
                served_model_name=str(
                    raw.get("served_model_name") or entry.get("served_model_name") or model
                ),
                api_key=str(raw["api_key"]) if raw.get("api_key") else (
                    str(entry["api_key"]) if entry.get("api_key") else None
                ),
                api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else (
                    str(entry["api_key_env"]) if entry.get("api_key_env") else None
                ),
                max_num_seqs=int(max_num_seqs) if max_num_seqs is not None else None,
            )
        )
    return replicas


def interleave_catalog_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin catalog rows by field while preserving order within each field."""
    order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        field = str(row.get("field") or "")
        if field not in buckets:
            order.append(field)
            buckets[field] = []
        buckets[field].append(row)

    result: list[dict[str, Any]] = []
    positions = {field: 0 for field in order}
    while len(result) < len(rows):
        for field in order:
            position = positions[field]
            bucket = buckets[field]
            if position < len(bucket):
                result.append(bucket[position])
                positions[field] = position + 1
    return result


def batch_config(config: dict[str, Any]) -> dict[str, Any]:
    value = table(config, "eval").get("batch", {})
    return value if isinstance(value, dict) else {}


def benchmark_rollout_n(config: dict[str, Any], tasks: list[str]) -> int:
    """Return the largest rollout count declared by the selected TOMLs."""

    specs = config.get("_benchmark_specs", {})
    if not isinstance(specs, dict):
        return 1
    largest = 1
    for task in tasks:
        canonical = str(task).split("|", 1)[0]
        if canonical.startswith("g1h__"):
            canonical = canonical[len("g1h__") :]
        spec = specs.get(canonical)
        if spec is None:
            parents = [
                (name, value)
                for name, value in specs.items()
                if canonical.startswith(f"{name}:")
            ]
            if parents:
                spec = max(parents, key=lambda item: len(item[0]))[1]
        evaluation = spec.get("evaluation", {}) if isinstance(spec, dict) else {}
        try:
            largest = max(largest, int(evaluation.get("rollout_n", 1)))
        except (AttributeError, TypeError, ValueError):
            continue
    return largest


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _stamp_from_iso(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace("+00:00", "Z")


def resolve_batch_plan(
    args: Any,
    config: dict[str, Any],
    *,
    lighteval_tasks_override: list[str] | None = None,
) -> list[BatchUnit]:
    batch = batch_config(config)
    models = _as_str_list(pick(getattr(args, "models", None), batch.get("models")))
    lighteval_tasks = list(lighteval_tasks_override or _as_str_list(getattr(args, "tasks", None)))
    fc_tasks = _as_str_list(getattr(args, "fc_tasks", None))
    if not lighteval_tasks and not fc_tasks:
        # No CLI benchmark selection: fall back to the config suite as a whole.
        lighteval_tasks = _as_str_list(batch.get("tasks"))
        fc_tasks = _as_str_list(batch.get("fc_tasks"))

    if not models:
        raise SystemExit("no models given: pass --models or set [eval.batch].models in the config")
    if not lighteval_tasks and not fc_tasks:
        raise SystemExit(
            "no benchmarks given: pass --tasks/--fc-tasks or set [eval.batch].tasks / "
            "[eval.batch].fc_tasks in the config"
        )

    # Interleave models by benchmark so one slow model cannot monopolize the
    # head of the queue. Every pair remains an independent persistence unit.
    units: list[BatchUnit] = []
    for task in lighteval_tasks:
        for model in models:
            units.append(BatchUnit(model=model, kind="lighteval", tasks=[task]))
    for task in fc_tasks:
        for model in models:
            units.append(BatchUnit(model=model, kind="fc", tasks=[task]))
    return units


def resolve_parallel_cap(args: Any, config: dict[str, Any]) -> int | None:
    value = pick(getattr(args, "parallel", None), batch_config(config).get("parallel"))
    if value is None:
        return None
    try:
        parallel = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit("[eval.batch].parallel/--parallel must be a positive integer") from error
    if parallel <= 0:
        raise SystemExit("[eval.batch].parallel/--parallel must be a positive integer")
    return parallel


def derive_postprocess_workers(
    *,
    runnable_count: int,
    score_capacity: int,
    configured_ceiling: int | None = None,
) -> int:
    """Size the non-model scoring pool from scheduler-visible capacity."""

    runnable_count = max(1, int(runnable_count))
    score_capacity = max(1, int(score_capacity))
    workers = score_capacity
    if configured_ceiling is not None:
        workers = min(workers, max(1, int(configured_ceiling)))
    return max(1, min(runnable_count, workers))


def _nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _nested_value(child, key)
            if found is not None:
                return found
    return None


def _metric_value(text: str, metric: str) -> int:
    values: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(metric) or stripped.startswith("#"):
            continue
        try:
            values.append(float(stripped.rsplit(None, 1)[-1]))
        except (IndexError, ValueError):
            continue
    return max(0, int(sum(values)))


def probe_model_runtime(
    *,
    base_url: str,
    api_key: str | None,
    timeout: float = 5.0,
) -> tuple[int | None, int, int, str]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    max_num_seqs: int | None = None
    source = "fallback"
    try:
        response = requests.get(
            f"{root}/server_info",
            params={"config_format": "json"},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        value = _nested_value(response.json(), "max_num_seqs")
        if value is not None and int(value) > 0:
            max_num_seqs = int(value)
            source = "server_info"
    except (OSError, TypeError, ValueError, requests.RequestException):
        pass
    running = waiting = 0
    try:
        response = requests.get(f"{root}/metrics", headers=headers, timeout=timeout)
        response.raise_for_status()
        running = _metric_value(response.text, "vllm:num_requests_running")
        waiting = _metric_value(response.text, "vllm:num_requests_waiting")
    except (OSError, requests.RequestException):
        pass
    return max_num_seqs, running, waiting, source


def derive_model_concurrency(
    *,
    model: str,
    pending_benchmarks: int,
    rollout_n: int,
    max_num_seqs: int | None,
    configured_request_ceiling: int | None,
    running_requests: int = 0,
    waiting_requests: int = 0,
    source: str = "fallback",
    open_file_limit: int | None = None,
    replica_count: int = 1,
) -> ModelConcurrency:
    rollout_n = max(1, int(rollout_n))
    pending_benchmarks = max(1, int(pending_benchmarks))
    benchmark_workers = min(pending_benchmarks, max(1, int(replica_count)))
    if max_num_seqs is None:
        return ModelConcurrency(
            model,
            benchmark_workers,
            1,
            rollout_n,
            None,
            running_requests,
            waiting_requests,
            source,
        )
    available_sequences = max(rollout_n, int(max_num_seqs) - max(0, int(running_requests)))
    request_slots = max(1, available_sequences // rollout_n)
    if waiting_requests > 0:
        request_slots = 1
    # Each endpoint replica is a scheduling resource. One (model, benchmark)
    # generation task exclusively owns one replica and receives that replica's
    # full request capacity until generation completes.
    concurrent_requests = request_slots
    if configured_request_ceiling is not None:
        concurrent_requests = min(concurrent_requests, max(1, int(configured_request_ceiling)))
    open_file_cap = request_cap_from_open_files(open_file_limit)
    if open_file_cap is not None and concurrent_requests > open_file_cap:
        concurrent_requests = open_file_cap
        source = f"{source}+nofile:{open_file_limit}"
    return ModelConcurrency(
        model,
        benchmark_workers,
        concurrent_requests,
        rollout_n,
        int(max_num_seqs),
        running_requests,
        waiting_requests,
        source,
    )


def soft_open_file_limit() -> int | None:
    """Return the process FD ceiling when the platform exposes RLIMIT_NOFILE."""

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            return None
        return max(0, int(soft))
    except (ImportError, OSError, ValueError):
        return None


def request_cap_from_open_files(open_file_limit: int | None) -> int | None:
    """Reserve descriptors for Python, HTTP, database, and dataset files."""

    if open_file_limit is None:
        return None
    reserve = max(OPEN_FILE_RESERVE_MIN, int(open_file_limit) // 8)
    usable = max(OPEN_FILES_PER_REQUEST, int(open_file_limit) - reserve)
    return max(1, usable // OPEN_FILES_PER_REQUEST)


def resolve_model_concurrency(
    *,
    model: str,
    pending_benchmarks: int,
    args: Any,
    config: dict[str, Any],
    env: dict[str, str],
    replicas: list[ModelReplica] | None = None,
    rollout_n_override: int | None = None,
) -> ModelConcurrency:
    lighteval = table(config, "lighteval")
    policy = lighteval.get("g1h") if isinstance(lighteval.get("g1h"), dict) else {}
    replicas = replicas or [ModelReplica(name="default")]
    rollout_n = int(
        rollout_n_override
        or policy.get("rollout_n")
        or policy.get("avg_k")
        or 1
    )
    # Presets describe evaluation semantics, not model capacity. Only an
    # explicit CLI/environment override may cap the model-derived request count.
    ceiling_value = pick(
        getattr(args, "concurrent_requests", None),
        env_value(env, "HELICOPTER_EVAL_CONCURRENT_REQUESTS"),
    )
    ceiling = int(ceiling_value) if ceiling_value is not None else None
    try:
        model_config = resolve_model_entry(config, model)
    except SystemExit:
        return derive_model_concurrency(
            model=model,
            pending_benchmarks=pending_benchmarks,
            rollout_n=rollout_n,
            max_num_seqs=None,
            configured_request_ceiling=ceiling,
            open_file_limit=soft_open_file_limit(),
            replica_count=len(replicas),
        )
    infer_config = model_config.get("infer") if isinstance(model_config.get("infer"), dict) else {}
    primary_replica = replicas[0]
    configured_max = (
        primary_replica.max_num_seqs
        or infer_config.get("max_num_seqs")
        or model_config.get("max_num_seqs")
    )
    max_num_seqs = int(configured_max) if configured_max is not None else None
    running = waiting = 0
    source = "model_config" if max_num_seqs is not None else "fallback"
    configured_endpoint = pick(
        getattr(args, "base_url", None),
        primary_replica.base_url,
        model_config.get("base_url"),
        env_value(env, "HELICOPTER_EVAL_BASE_URL", "OPENAI_BASE_URL"),
        lighteval.get("base_url"),
    )
    if configured_endpoint:
        unit_args = copy.copy(args)
        unit_args.model = model
        if primary_replica.base_url:
            unit_args.base_url = primary_replica.base_url
        base_url = local_openai_base_url(config, env, unit_args)
        replica_api_key = primary_replica.api_key
        if not replica_api_key and primary_replica.api_key_env:
            replica_api_key = env.get(primary_replica.api_key_env)
        api_key = pick(
            getattr(args, "api_key", None),
            replica_api_key,
            model_config.get("api_key"),
            env_value(env, "HELICOPTER_EVAL_API_KEY", "OPENAI_API_KEY"),
            lighteval.get("api_key"),
        )
        probed_max, running, waiting, probed_source = probe_model_runtime(
            base_url=base_url,
            api_key=str(api_key) if api_key else None,
        )
        if probed_max is not None:
            max_num_seqs = probed_max
            source = probed_source
    return derive_model_concurrency(
        model=model,
        pending_benchmarks=pending_benchmarks,
        rollout_n=rollout_n,
        max_num_seqs=max_num_seqs,
        configured_request_ceiling=ceiling,
        running_requests=running,
        waiting_requests=waiting,
        source=f"{source}+replicas:{len(replicas)}",
        open_file_limit=soft_open_file_limit(),
        replica_count=len(replicas),
    )


def detect_idle_gpus(*, threshold_mib: float = DEFAULT_GPU_IDLE_MIB) -> list[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[int] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            used = float(parts[1])
        except ValueError:
            continue
        if used < threshold_mib:
            gpus.append(int(parts[0]))
    return gpus


def resolve_slots(args: Any, config: dict[str, Any], env: dict[str, str]) -> list[GpuSlot]:
    """Build the GPU slots the batch may run on.

    With --no-server or an explicit --base-url the endpoint is external, so
    GPU pinning and per-slot ports do not apply: one generic slot is returned.
    """
    if getattr(args, "no_server", False) or getattr(args, "base_url", None):
        return [GpuSlot(index=0, gpu=None, port=0)]
    for model in _as_str_list(getattr(args, "models", None) or batch_config(config).get("models")):
        try:
            if any(replica.base_url for replica in resolve_model_replicas(config, model)):
                return [GpuSlot(index=0, gpu=None, port=0)]
        except SystemExit:
            continue

    port_base = int(pick(getattr(args, "port_base", None), batch_config(config).get("port_base"), DEFAULT_PORT_BASE))
    explicit = _as_str_list(getattr(args, "gpus", None) or batch_config(config).get("gpus"))
    if explicit:
        gpus = [int(item) for item in explicit]
    else:
        gpus = detect_idle_gpus(
            threshold_mib=float(
                pick(
                    getattr(args, "gpu_idle_max_mem", None),
                    batch_config(config).get("gpu_idle_max_mem"),
                    DEFAULT_GPU_IDLE_MIB,
                )
            )
        )
    if not gpus:
        # No visible GPU management available: single slot on the default port.
        return [GpuSlot(index=0, gpu=None, port=port_base)]
    return [GpuSlot(index=i, gpu=gpu, port=port_base + i) for i, gpu in enumerate(gpus)]


async def _query_completed_datasets(
    *,
    model_name: str,
    datasets: list[str],
    root: Path,
    identities: dict[str, tuple[str, dict[str, Any]]] | None = None,
) -> set[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.cores.normalize import normalize_model_name, split_dataset
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.models import Score
    from scoreboard_server.db.settings import DatabaseSettings

    await init_db(DatabaseSettings.from_env(), generate_schemas=False)
    completed: set[str] = set()
    try:
        normalized_model = normalize_model_name(model_name)
        for dataset in datasets:
            exact_query = Score.filter(
                task__model__model_name=normalized_model,
                task__benchmark__benchmark_name=dataset,
                task__benchmark__benchmark_split="",
                task__is_tmp=False,
                task__status="Completed",
            )
            identity = (identities or {}).get(dataset)
            if identity is not None:
                config_path, sampling_config = identity
                exact_query = exact_query.filter(task__config_path=config_path)
                exact_scores = await exact_query.prefetch_related("task")
                exact_exists = any(
                    json.dumps(score.task.sampling_config, sort_keys=True, separators=(",", ":"))
                    == json.dumps(sampling_config, sort_keys=True, separators=(",", ":"))
                    for score in exact_scores
                )
            else:
                exact_exists = await exact_query.exists()
            if exact_exists:
                completed.add(dataset)
                continue
            name, split = split_dataset(dataset)
            split_query = Score.filter(
                task__model__model_name=normalized_model,
                task__benchmark__benchmark_name=name,
                task__benchmark__benchmark_split=split,
                task__is_tmp=False,
                task__status="Completed",
            )
            if identity is not None:
                config_path, sampling_config = identity
                split_query = split_query.filter(task__config_path=config_path)
                split_scores = await split_query.prefetch_related("task")
                exists = any(
                    json.dumps(score.task.sampling_config, sort_keys=True, separators=(",", ":"))
                    == json.dumps(sampling_config, sort_keys=True, separators=(",", ":"))
                    for score in split_scores
                )
            else:
                exists = await split_query.exists()
            if exists:
                completed.add(dataset)
    finally:
        await close_db()
    return completed


def unit_dataset_names(unit: BatchUnit) -> dict[str, str]:
    """Map each task entry of a unit to its scoreboard benchmark name."""
    if unit.kind == "lighteval":
        return {task: scoreboard_dataset_name(task) for task in unit.tasks}
    return {task: task for task in unit.tasks}


def lighteval_scoreboard_identities(
    unit: BatchUnit,
    *,
    args: Any,
    config: dict[str, Any],
    env: dict[str, str],
    root: Path,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Resolve the same config identity used when a LightEval task is stored."""

    from .scoreboard_bridge import sampling_config_from_env

    identities: dict[str, tuple[str, dict[str, Any]]] = {}
    for task in unit.tasks:
        task_unit = BatchUnit(model=unit.model, kind=unit.kind, tasks=[task])
        task_args = _unit_args(args, task_unit, GpuSlot(index=0, gpu=None, port=0))
        plan = build_lighteval_plan(task_args, root=root, env=env, config=config)
        config_path = plan.env.get("HELICOPTER_SCOREBOARD_CONFIG_PATH", "")
        if not config_path:
            raise RuntimeError("LightEval plan did not resolve a scoreboard config path")
        identities[scoreboard_dataset_name(task)] = (
            config_path,
            sampling_config_from_env(plan.env),
        )
    return identities


def filter_completed_units(
    units: list[BatchUnit],
    *,
    args: Any,
    config: dict[str, Any],
    env: dict[str, str],
    root: Path,
) -> None:
    """Drop tasks that already have a scoreboard score; mutates units in place."""
    for unit in units:
        mapping = unit_dataset_names(unit)
        unit_args = copy.copy(args)
        unit_args.model = unit.model
        model_name = scoreboard_model_name(unit_args, config)
        identities = (
            lighteval_scoreboard_identities(
                unit,
                args=args,
                config=config,
                env=env,
                root=root,
            )
            if unit.kind == "lighteval"
            else None
        )
        try:
            with SCOREBOARD_LOCK, _scoreboard_env(env):
                completed = asyncio.run(
                    _query_completed_datasets(
                        model_name=model_name,
                        datasets=sorted(set(mapping.values())),
                        root=root,
                        identities=identities,
                    )
                )
        except Exception as error:  # noqa: BLE001 - fall back to running everything
            print(f"eval batch: skip-completed check failed ({error}); running all benchmarks")
            return
        remaining = [task for task in unit.tasks if mapping[task] not in completed]
        unit.skipped_tasks = [task for task in unit.tasks if mapping[task] in completed]
        unit.tasks = remaining
        if not remaining:
            unit.status = "skipped"
            unit.message = "all benchmarks already scored"


async def _query_catalog_lighteval_tasks(
    *,
    root: Path,
    env: dict[str, str],
    scope: str,
    fields: list[str],
    limit: int | None,
) -> list[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        rows = await store.list_benchmark_catalog(
            scope=scope,
            fields=fields or None,
            source=CATALOG_SOURCE,
            target_kind=CATALOG_TARGET_KIND,
            run_status=CATALOG_RUN_STATUS,
            limit=None,
        )
    finally:
        await close_db()
    rows = interleave_catalog_rows(rows)
    if limit is not None and int(limit) > 0:
        rows = rows[: int(limit)]
    return [str(row["name"]) for row in rows]


def query_catalog_lighteval_tasks(*, args: Any, root: Path, env: dict[str, str]) -> list[str]:
    scope = str(
        pick(
            getattr(args, "benchmark_scope", None),
            CATALOG_SCOPE,
        )
    )
    fields = _as_str_list(getattr(args, "benchmark_fields", None))
    limit = getattr(args, "benchmark_limit", None)
    with SCOREBOARD_LOCK, _scoreboard_env(env):
        tasks = asyncio.run(
            _query_catalog_lighteval_tasks(
                root=root,
                env=env,
                scope=scope,
                fields=fields,
                limit=int(limit) if limit else None,
            )
        )
    if not tasks:
        raise SystemExit(f"no LightEval tasks found in benchmark_catalog for scope={scope!r}")
    return tasks


def _safe_unit_slug(unit: BatchUnit, slot: GpuSlot) -> str:
    digest = hashlib.sha1(",".join(unit.tasks).encode("utf-8")).hexdigest()[:8]
    raw = f"slot{slot.index:02d}_gpu{slot.gpu if slot.gpu is not None else 'none'}_{unit.model}_{unit.kind}_{digest}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or f"unit_{digest}"


def _unit_args(
    args: Any,
    unit: BatchUnit,
    slot: GpuSlot,
    *,
    replica: ModelReplica | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    unit_args = copy.copy(args)
    unit_args.model = unit.model
    unit_args.tasks = ",".join(unit.tasks)
    unit_args.no_server = getattr(args, "no_server", False)
    unit_args.keep_server = False
    unit_args.scoreboard = True
    if replica is not None:
        if replica.base_url:
            unit_args.base_url = replica.base_url
            unit_args.no_server = True
        if replica.served_model_name:
            unit_args.lighteval_model_name = replica.served_model_name
        if replica.api_key:
            unit_args.api_key = replica.api_key
        elif replica.api_key_env:
            api_key = (env or {}).get(replica.api_key_env)
            if not api_key:
                raise SystemExit(
                    f"endpoint replica {replica.name!r} requires environment variable {replica.api_key_env}"
                )
            unit_args.api_key = api_key
    elif slot.port:
        unit_args.base_url = f"http://127.0.0.1:{slot.port}/v1"
    batch_dir = getattr(args, "_batch_run_dir", None) or getattr(args, "output_dir", None)
    if batch_dir:
        unit_dir = Path(str(batch_dir)) / _safe_unit_slug(unit, slot)
        unit_args.output_dir = str(unit_dir / ("lighteval" if unit.kind == "lighteval" else "function_calling"))
    return unit_args


def _unit_env(env: dict[str, str], slot: GpuSlot) -> dict[str, str]:
    if slot.gpu is None:
        return dict(env)
    slot_env = dict(env)
    slot_env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
    return slot_env


def run_unit(
    unit: BatchUnit,
    *,
    args: Any,
    slot: GpuSlot,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
    max_retries: int,
    concurrent_requests: int | None = None,
    pipeline_stage: str | None = None,
    extra_env: dict[str, str] | None = None,
    replica: ModelReplica | None = None,
) -> None:
    stage_note = f"/{pipeline_stage}" if pipeline_stage else ""
    label = f"{unit.model}/{unit.kind}{stage_note}"
    unit_args = _unit_args(args, unit, slot, replica=replica, env=env)
    if concurrent_requests is not None:
        unit_args.concurrent_requests = int(concurrent_requests)
    unit_env = _unit_env(env, slot)
    if pipeline_stage == "score":
        unit_args.no_server = True
    if pipeline_stage:
        unit_env["HELICOPTER_PIPELINE_STAGE"] = pipeline_stage
    if extra_env:
        unit_env.update({str(key): str(value) for key, value in extra_env.items()})
    runner = run_eval if unit.kind == "lighteval" else run_function_calling_eval
    started = time.monotonic()
    unit.started_at = _utc_now()
    if replica is not None:
        unit.replica = replica.name
    unit.slot_index = slot.index
    unit.gpu = slot.gpu
    unit.port = slot.port or None
    for attempt in range(1, max_retries + 2):
        unit.attempts = attempt
        gpu_note = f" gpu={slot.gpu}" if slot.gpu is not None else ""
        print(f"eval batch: [{label}] attempt {attempt}{gpu_note} tasks={unit_args.tasks}")
        try:
            exit_code = runner(unit_args, root=root, env=unit_env, config=config)
        except SystemExit as error:
            exit_code = 1
            unit.message = str(error)
        except Exception as error:  # noqa: BLE001 - one unit must not kill the batch
            exit_code = 1
            unit.message = f"{type(error).__name__}: {error}"
        unit.exit_code = exit_code
        if exit_code == 0:
            unit.status = "completed"
            unit.message = ""
            break
        unit.status = "failed"
        if not unit.message:
            unit.message = f"exit code {exit_code}"
        if attempt <= max_retries:
            print(f"eval batch: [{label}] failed ({unit.message}); retrying")
    unit.elapsed_seconds = time.monotonic() - started
    unit.ended_at = _utc_now()


def format_summary(units: list[BatchUnit]) -> str:
    lines = ["model\treplica\tkind\tstatus\tattempts\telapsed_s\ttasks\tskipped\tmessage"]
    for unit in units:
        lines.append(
            "\t".join(
                [
                    unit.model,
                    unit.replica or "-",
                    unit.kind,
                    unit.status,
                    str(unit.attempts),
                    f"{unit.elapsed_seconds:.0f}",
                    ",".join(unit.tasks) or "-",
                    ",".join(unit.skipped_tasks) or "-",
                    unit.message or "-",
                ]
            )
        )
    return "\n".join(lines)


def _unit_record(unit: BatchUnit) -> dict[str, Any]:
    return {
        "model": unit.model,
        "replica": unit.replica,
        "kind": unit.kind,
        "tasks": list(unit.tasks),
        "skipped_tasks": list(unit.skipped_tasks),
        "status": unit.status,
        "message": unit.message,
        "attempts": unit.attempts,
        "elapsed_seconds": unit.elapsed_seconds,
        "slot_index": unit.slot_index,
        "gpu": unit.gpu,
        "port": unit.port,
        "exit_code": unit.exit_code,
        "started_at": unit.started_at,
        "ended_at": unit.ended_at,
    }


def _slot_record(slot: GpuSlot) -> dict[str, Any]:
    return {"index": slot.index, "gpu": slot.gpu, "port": slot.port or None}


def _status_counts(units: list[BatchUnit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit.status] = counts.get(unit.status, 0) + 1
    return counts


def resolve_batch_report_path(args: Any, config: dict[str, Any], root: Path, *, started_at: str) -> Path | None:
    batch = batch_config(config)
    configured = pick(getattr(args, "batch_output", None), batch.get("output"))
    if configured:
        path = Path(str(configured))
    else:
        return None
    if not path.is_absolute():
        path = root / path
    return path


def write_batch_report(
    *,
    path: Path | None,
    units: list[BatchUnit],
    slots: list[GpuSlot],
    args: Any,
    exit_code: int,
    started_at: str,
    ended_at: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": exit_code,
        "summary": {
            "units": len(units),
            "status_counts": _status_counts(units),
        },
        "selection": {
            "models": _as_str_list(getattr(args, "models", None)),
            "tasks": _as_str_list(getattr(args, "tasks", None)),
            "fc_tasks": _as_str_list(getattr(args, "fc_tasks", None)),
            "scoreboard": bool(getattr(args, "scoreboard", False)),
            "rerun": bool(getattr(args, "rerun", False)),
            "dry_run": bool(getattr(args, "dry_run", False)),
        },
        "runtime": {
            "parallel": getattr(args, "parallel", None),
            "max_retries": getattr(args, "max_retries", None),
            "no_server": bool(getattr(args, "no_server", False)),
            "base_url": getattr(args, "base_url", None),
            "port_base": getattr(args, "port_base", None),
            "wkv_mode": getattr(args, "wkv_mode", None),
            "emb_device": getattr(args, "emb_device", None),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
            "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", None),
            "max_num_seqs": getattr(args, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(args, "max_num_batched_tokens", None),
            "vllm_env": getattr(args, "vllm_env", None),
        },
        "slots": [_slot_record(slot) for slot in slots],
        "units": [_unit_record(unit) for unit in units],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"eval batch: wrote report {path}")


def run_batch(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    started_at = _utc_now()
    report_path: Path | None = None
    slots = resolve_slots(args, config, env)
    report_path = resolve_batch_report_path(args, config, root, started_at=started_at)
    batch_run_dir = (
        report_path.with_suffix("")
        if report_path is not None
        else root / "results/eval_batch" / f"dry_run_{_stamp_from_iso(started_at)}"
    )
    setattr(args, "_batch_run_dir", str(batch_run_dir))

    lighteval_tasks_override = None
    if getattr(args, "tasks_from_db", False):
        if getattr(args, "tasks", None):
            raise SystemExit("--tasks-from-db cannot be combined with --tasks")
        lighteval_tasks_override = query_catalog_lighteval_tasks(args=args, root=root, env=env)
    units = resolve_batch_plan(args, config, lighteval_tasks_override=lighteval_tasks_override)

    skip_completed = bool(getattr(args, "scoreboard", False)) and not getattr(args, "rerun", False)
    if skip_completed:
        filter_completed_units(units, args=args, config=config, env=env, root=root)

    runnable = [unit for unit in units if unit.status == "pending"]
    if not runnable:
        print("eval batch: nothing to run (all benchmarks already scored)")
        print(format_summary(units))
        write_batch_report(
            path=report_path,
            units=units,
            slots=slots,
            args=args,
            exit_code=0,
            started_at=started_at,
            ended_at=_utc_now(),
        )
        return 0

    pending_by_model = {
        model: sum(1 for unit in runnable if unit.model == model)
        for model in {unit.model for unit in runnable}
    }
    model_replicas: dict[str, list[ModelReplica]] = {}
    for model in pending_by_model:
        if getattr(args, "base_url", None):
            model_replicas[model] = [
                ModelReplica(name="cli", base_url=str(args.base_url))
            ]
        else:
            try:
                model_replicas[model] = resolve_model_replicas(config, model)
            except SystemExit:
                # Preserve per-unit failure reporting for unknown model aliases.
                model_replicas[model] = [ModelReplica(name="default")]
    model_concurrency = {
        model: resolve_model_concurrency(
            model=model,
            pending_benchmarks=count,
            args=args,
            config=config,
            env=env,
            replicas=model_replicas[model],
            rollout_n_override=benchmark_rollout_n(
                config,
                [
                    task
                    for unit in runnable
                    if unit.model == model
                    for task in unit.tasks
                ],
            ),
        )
        for model, count in pending_by_model.items()
    }
    if getattr(args, "dry_run", False):
        print(
            f"eval batch: {len(runnable)} unit(s): "
            + "; ".join(f"{unit.model}/{unit.kind}:{','.join(unit.tasks)}" for unit in runnable)
        )
        for model in sorted(model_concurrency):
            item = model_concurrency[model]
            print(
                f"eval batch: model={model} workers={item.benchmark_workers} "
                f"concurrent_requests={item.concurrent_requests} rollout_n={item.rollout_n} "
                f"max_num_seqs={item.max_num_seqs or 'unknown'} source={item.source}"
            )
        dry_run_positions = {model: 0 for model in model_replicas}
        for unit in runnable:
            replicas = model_replicas[unit.model]
            position = dry_run_positions[unit.model]
            replica = replicas[position % len(replicas)]
            dry_run_positions[unit.model] = position + 1
            run_unit(
                unit,
                args=args,
                slot=slots[0],
                root=root,
                env=env,
                config=config,
                max_retries=0,
                concurrent_requests=model_concurrency[unit.model].concurrent_requests,
                replica=replica,
            )
            if unit.status == "completed":
                unit.status = "dry_run"
                unit.message = "plan only"
        failed = [unit for unit in units if unit.status == "failed"]
        exit_code = 1 if failed else 0
        write_batch_report(
            path=report_path,
            units=units,
            slots=slots,
            args=args,
            exit_code=exit_code,
            started_at=started_at,
            ended_at=_utc_now(),
        )
        return exit_code

    external_endpoint = bool(
        getattr(args, "no_server", False)
        or getattr(args, "base_url", None)
        or any(replica.base_url for replicas in model_replicas.values() for replica in replicas)
    )
    workers = sum(item.benchmark_workers for item in model_concurrency.values())
    if not external_endpoint:
        workers = min(workers, len(slots))
    parallel_cap = resolve_parallel_cap(args, config)
    if parallel_cap is not None:
        workers = min(workers, parallel_cap)
    workers = max(1, min(workers, len(runnable)))
    max_retries = max(0, int(getattr(args, "max_retries", None) or 0))
    print(
        f"eval batch: running {len(runnable)} unit(s) with {workers} model worker(s)"
    )
    for model in sorted(model_concurrency):
        item = model_concurrency[model]
        print(
            f"eval batch: model={model} workers={item.benchmark_workers} "
            f"concurrent_requests={item.concurrent_requests} rollout_n={item.rollout_n} "
            f"max_num_seqs={item.max_num_seqs or 'unknown'} "
            f"running={item.running_requests} waiting={item.waiting_requests} source={item.source}"
        )

    slot_queue: queue.Queue[GpuSlot] = queue.Queue()
    for slot in slots:
        slot_queue.put(slot)

    replica_queues: dict[str, queue.Queue[ModelReplica]] = {}
    for model, replicas in model_replicas.items():
        replica_queue: queue.Queue[ModelReplica] = queue.Queue()
        for replica in replicas:
            replica_queue.put(replica)
        replica_queues[model] = replica_queue
    unit_indices = {id(unit): index for index, unit in enumerate(runnable)}
    score_capacity = max(1, int(env.get("HELICOPTER_SCORE_CONCURRENT_REQUESTS", "10")))
    configured_score_workers = env.get("HELICOPTER_SCORE_WORKERS")
    score_workers = derive_postprocess_workers(
        runnable_count=len(runnable),
        score_capacity=score_capacity,
        configured_ceiling=int(configured_score_workers) if configured_score_workers else None,
    )
    score_executor = ThreadPoolExecutor(max_workers=score_workers)
    score_futures: list[Any] = []
    score_futures_lock = threading.Lock()
    print(f"eval batch: LightEval postprocess workers={score_workers}")

    def score_worker(unit: BatchUnit) -> None:
        run_unit(
            unit,
            args=args,
            slot=GpuSlot(index=unit_indices[id(unit)], gpu=None, port=0),
            root=root,
            env=env,
            config=config,
            max_retries=max_retries,
            concurrent_requests=1,
            pipeline_stage="score",
        )

    def worker(unit: BatchUnit) -> None:
        replica = replica_queues[unit.model].get()
        slot = (
            GpuSlot(index=unit_indices[id(unit)], gpu=None, port=0)
            if external_endpoint
            else slot_queue.get()
        )
        try:
            run_unit(
                unit,
                args=args,
                slot=slot,
                root=root,
                env=env,
                config=config,
                max_retries=max_retries,
                concurrent_requests=model_concurrency[unit.model].concurrent_requests,
                pipeline_stage="generate",
                replica=replica,
            )
        finally:
            if not external_endpoint:
                slot_queue.put(slot)
            replica_queues[unit.model].put(replica)

        if unit.status == "completed":
            with score_futures_lock:
                score_futures.append(score_executor.submit(score_worker, unit))

    try:
        run_model_aware_scheduler(
            runnable,
            model_worker_limits={
                model: item.benchmark_workers
                for model, item in model_concurrency.items()
            },
            max_workers=workers,
            worker=worker,
        )
        for future in score_futures:
            future.result()
    finally:
        score_executor.shutdown(wait=True)

    print(format_summary(units))
    failed = [unit for unit in units if unit.status == "failed"]
    exit_code = 1 if failed else 0
    write_batch_report(
        path=report_path,
        units=units,
        slots=slots,
        args=args,
        exit_code=exit_code,
        started_at=started_at,
        ended_at=_utc_now(),
    )
    return exit_code
