from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .commands import (
    CommandPlan,
    build_infer_plan,
    build_lighteval_plan,
    is_local_base_url,
    local_openai_base_url,
)
from .config import resolve_model_entry, table
from .env import pick
from .rwkv_config import canonical_task_name
from .performance import (
    base_url_from_lighteval_command,
    derive_metrics_url,
    output_dir_from_command,
    run_lighteval_with_performance,
)


DEFAULT_SERVER_TIMEOUT_S = 600.0
SERVER_POLL_INTERVAL_S = 2.0


def resolve_run_tasks(args: Any, config: dict[str, Any]) -> str:
    lighteval = table(config, "lighteval")
    configured = lighteval.get("tasks")
    if isinstance(configured, list):
        configured = ",".join(str(item) for item in configured if str(item))
    tasks = pick(getattr(args, "tasks", None), configured)
    if not tasks:
        raise SystemExit(
            "no tasks given: pass a task string (e.g. 'gsm8k|0') or set [lighteval].tasks in the config"
        )
    return str(tasks)


def health_url_for(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def server_is_healthy(base_url: str, *, timeout_s: float = 2.0) -> bool:
    try:
        with urlopen(health_url_for(base_url), timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def endpoint_model_ids(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 5.0,
) -> tuple[str, ...]:
    """Return the model IDs advertised by an OpenAI-compatible endpoint."""

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(health_url_for(base_url), headers=headers)
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return ()
    return tuple(
        str(row.get("id"))
        for row in rows
        if isinstance(row, dict) and row.get("id")
    )


def validate_endpoint_model(
    base_url: str,
    expected_model: str | None,
    *,
    api_key: str | None = None,
    allow_mismatch: bool = False,
) -> tuple[str, ...]:
    """Reject a live endpoint whose advertised model differs from the task model.

    A wrong ``--base-url`` otherwise looks healthy and silently writes valid-looking
    scores for the wrong model.  The explicit escape hatch is retained for proxy
    endpoints that intentionally hide or rewrite the served model ID.
    """

    expected = str(expected_model or "").strip()
    if not expected:
        return ()
    try:
        model_ids = endpoint_model_ids(base_url, api_key=api_key)
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"model endpoint check failed at {health_url_for(base_url)}: {error}"
        ) from error
    if not model_ids:
        raise SystemExit(
            f"model endpoint check returned no model IDs at {health_url_for(base_url)}"
        )
    expected_names = {expected, expected.rsplit("/", 1)[-1]}
    if any(model_id in expected_names for model_id in model_ids):
        return model_ids
    message = (
        f"model endpoint mismatch at {base_url}: requested {expected!r}, "
        f"served {', '.join(model_ids)}; use the model's configured endpoint"
    )
    if allow_mismatch:
        print(f"WARNING: {message} (HELICOPTER_ALLOW_MODEL_MISMATCH enabled)")
        return model_ids
    raise SystemExit(message)


def _tail_lines(path: Path, count: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def wait_for_server(
    base_url: str,
    *,
    process: subprocess.Popen[bytes],
    log_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"vLLM server exited early with code {process.returncode}; last log lines from {log_path}:\n"
                f"{_tail_lines(log_path)}"
            )
        if server_is_healthy(base_url):
            return
        time.sleep(SERVER_POLL_INTERVAL_S)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    raise SystemExit(
        f"vLLM server did not become healthy within {int(timeout_s)}s at {health_url_for(base_url)}; "
        f"last log lines from {log_path}:\n{_tail_lines(log_path)}"
    )


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def infer_args_namespace(args: Any, *, port: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        dry_run=getattr(args, "dry_run", False),
        wkv_mode=getattr(args, "wkv_mode", None),
        emb_device=getattr(args, "emb_device", None),
        host=None,
        port=port,
        served_model_name=getattr(args, "lighteval_model_name", None),
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
        max_model_len=None,
        max_num_seqs=getattr(args, "max_num_seqs", None),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", None),
        enable_auto_tool_choice=getattr(args, "enable_auto_tool_choice", None),
        tool_call_parser=getattr(args, "tool_call_parser", None),
        vllm_env=getattr(args, "vllm_env", None),
    )


def port_from_base_url(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    return str(parsed.port) if parsed.port else None


def format_plan_for_display(plan: CommandPlan) -> str:
    pieces: list[str] = []
    if plan.cwd is not None:
        pieces.extend(["cd", shlex.quote(str(plan.cwd)), "&&"])
    if plan.shown_env:
        pieces.append("env")
        for key in sorted(plan.shown_env):
            pieces.append(f"{key}={shlex.quote(plan.shown_env[key])}")
    pieces.extend(shlex.quote(item) for item in plan.command)
    return " ".join(pieces)


def scoreboard_dataset_name(task_name: str) -> str:
    """Normalize a LightEval results key (``gsm8k|0`` or ``suite|task|0``) to a benchmark name."""
    parts = [part.strip() for part in str(task_name).split("|") if part.strip()]
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return canonical_task_name(parts[-1] if parts else str(task_name))


def scoreboard_model_name(args: Any, config: dict[str, Any]) -> str:
    model = resolve_model_entry(config, args.model)
    return str(
        pick(
            getattr(args, "lighteval_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )


def _first_nonempty_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [str(item) for item in value if item not in (None, "")]
        if len(texts) == 1:
            return texts[0]
        if texts:
            return json.dumps(texts, ensure_ascii=False, default=str)
    if value not in (None, ""):
        return str(value)
    return ""


def _lighteval_answer(model_response: Mapping[str, Any]) -> str:
    for key in ("text_post_processed", "text"):
        answer = _first_nonempty_text(model_response.get(key))
        if answer:
            return answer
    return json.dumps(dict(model_response), ensure_ascii=False, sort_keys=True, default=str)


def _lighteval_reference(doc: Mapping[str, Any]) -> str:
    choices = doc.get("choices")
    gold_indices = doc.get("gold_index")
    if isinstance(choices, list) and isinstance(gold_indices, list):
        selected = [choices[index] for index in gold_indices if isinstance(index, int) and 0 <= index < len(choices)]
        if selected:
            return _first_nonempty_text(selected)
    for key in ("expected_answer", "reference_answer", "solution", "answer", "target"):
        value = doc.get(key)
        if value not in (None, ""):
            return _first_nonempty_text(value)
    specific = doc.get("specific")
    if isinstance(specific, Mapping):
        for key in ("expected_answer", "reference_answer", "solution", "answer", "target"):
            value = specific.get(key)
            if value not in (None, ""):
                return _first_nonempty_text(value)
    return ""


def _lighteval_passed(metrics: Mapping[str, Any]) -> bool:
    values: list[float] = []
    for name, value in metrics.items():
        if "stderr" in str(name).lower() or isinstance(value, bool):
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return any(value > 0.0 for value in values)


def _lighteval_detail_payloads(
    *, detail_files: list[str], task_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import pandas as pd
    except ImportError:
        return [], []

    from .lighteval_answer_adapters import adapt_answer, answers_match
    from .scoreboard_bridge import _compact_lighteval_doc, _reference

    request_policy: Mapping[str, Any] = {}
    try:
        payload = json.loads(
            os.environ.get(
                "HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY",
                os.environ.get("HELICOPTER_LIGHTEEVAL_TASK_REQUEST_POLICY", ""),
            )
        )
        tasks = payload.get("tasks", {})
        if isinstance(tasks, Mapping) and len(tasks) == 1:
            entry = next(iter(tasks.values()))
            if isinstance(entry, Mapping):
                request_policy = entry
    except (TypeError, ValueError, StopIteration):
        pass

    completion_payloads: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    for item in detail_files:
        path = Path(item)
        try:
            rows = pd.read_parquet(path, columns=["doc", "metric", "model_response"]).to_dict("records")
        except Exception:  # noqa: BLE001 - skip unrelated or malformed LightEval detail files.
            continue
        for row in rows:
            doc = row.get("doc") if isinstance(row.get("doc"), Mapping) else {}
            row_task = str(doc.get("task_name") or "")
            if row_task and row_task != task_name:
                continue
            metrics = row.get("metric") if isinstance(row.get("metric"), Mapping) else {}
            response = row.get("model_response") if isinstance(row.get("model_response"), Mapping) else {}
            sample_index = len(completion_payloads)
            answer = adapt_answer(
                _lighteval_answer(response),
                domain=request_policy.get("domain"),
                request_format=request_policy.get("format"),
                prompt=str(response.get("input") or doc.get("query") or ""),
                stops=(request_policy.get("stop") if isinstance(request_policy.get("stop"), list) else []),
            )
            reference = adapt_answer(
                _reference(doc),
                domain=request_policy.get("domain"),
                request_format=request_policy.get("format"),
                prompt=str(response.get("input") or doc.get("query") or ""),
                stops=(request_policy.get("stop") if isinstance(request_policy.get("stop"), list) else []),
            )
            adapted_passed = answers_match(
                answer,
                reference,
                domain=request_policy.get("domain"),
                request_format=request_policy.get("format"),
            )
            if adapted_passed is not None:
                passed = adapted_passed
            elif any(
                any(token in str(name).lower() for token in ("avg@", "maj@", "gpass@", "stderr"))
                for name in metrics
            ):
                passed = False
            else:
                passed = _lighteval_passed(metrics)
            key = {"sample_index": sample_index, "repeat_index": 0, "pass_index": 0}
            completion_payloads.append(
                {
                    **key,
                    "prompt1": response.get("input") or doc.get("query") or "",
                    "completion1": answer,
                    "stop_reason1": None,
                    "stats": {"metrics": dict(metrics), "lighteval_task": task_name},
                    "agent_result": {
                        "doc": _compact_lighteval_doc(doc),
                        "model_response": dict(response),
                    },
                    "task_id": doc.get("id"),
                }
            )
            eval_payloads.append(
                {
                    **key,
                    "answer": answer,
                    "ref_answer": reference,
                    "raw_record": dict(doc),
                    "is_passed": passed,
                    "fail_reason": "" if passed else json.dumps(dict(metrics), ensure_ascii=False, default=str),
                }
            )
    return completion_payloads, eval_payloads


async def _ingest_scoreboard_results(
    *,
    result_files: list[str],
    detail_files: list[str] | None = None,
    model_name: str,
    root: Path,
    job_name: str = "lighteval",
) -> list[str]:
    scoreboard_path = root / "src/scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))

    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    recorded: list[str] = []
    try:
        store = ScoreboardStore(settings=settings)
        for item in result_files:
            path = Path(item)
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            results = payload.get("results")
            if not isinstance(results, dict):
                continue
            for task_name, metrics in results.items():
                if task_name == "all" or not isinstance(metrics, dict):
                    continue
                dataset = scoreboard_dataset_name(task_name)
                completion_payloads, eval_payloads = _lighteval_detail_payloads(
                    detail_files=list(detail_files or []),
                    task_name=str(task_name),
                )
                task_id, _inserted = await store.insert_completion_payloads_with_task(
                    payloads=completion_payloads,
                    task_id=None,
                    job_name=job_name,
                    dataset=dataset,
                    model=model_name,
                    is_param_search=False,
                    allow_resume=False,
                    num_samples=len(completion_payloads),
                )
                if task_id is None:
                    continue
                await store.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
                await store.record_score_payload(
                    task_id=task_id,
                    payload={"cot_mode": "NoCoT", "metrics": metrics},
                )
                recorded.append(f"{dataset} -> task {task_id}")
    finally:
        await close_db()
    return recorded


SCOREBOARD_ENV_PREFIXES = ("SCOREBOARD_DB_", "PG")

# Serializes scoreboard database access (Tortoise keeps global connection
# state, so concurrent batch workers must not init/close it simultaneously).
SCOREBOARD_LOCK = threading.Lock()


@contextmanager
def _scoreboard_env(env: dict[str, str]):
    """Expose dotenv-loaded database settings to DatabaseSettings.from_env()."""
    applied: dict[str, str | None] = {}
    for key, value in env.items():
        if key.startswith(SCOREBOARD_ENV_PREFIXES) and os.environ.get(key) != value:
            applied[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        yield
    finally:
        for key, previous in applied.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def ingest_scoreboard_results(
    *,
    result_files: list[str],
    detail_files: list[str] | None = None,
    model_name: str,
    root: Path,
    env: dict[str, str],
    job_name: str = "lighteval",
) -> None:
    if not result_files:
        print("eval run: no result files found; nothing to ingest into the scoreboard")
        return
    try:
        with SCOREBOARD_LOCK, _scoreboard_env(env):
            recorded = asyncio.run(
                _ingest_scoreboard_results(
                    result_files=result_files,
                    detail_files=detail_files,
                    model_name=model_name,
                    root=root,
                    job_name=job_name,
                )
            )
    except Exception as error:  # noqa: BLE001 - scoreboard ingestion must not fail the eval run
        print(f"eval run: scoreboard ingestion failed (results are still on disk): {error}")
        return
    for line in recorded:
        print(f"eval run: scoreboard score recorded: {line}")


def _pinned_scoreboard_task_id(env: dict[str, str]) -> str | None:
    pipeline_stage = str(env.get("HELICOPTER_PIPELINE_STAGE") or "").strip().lower()
    if pipeline_stage != "score":
        return None
    return str(env.get("HELICOPTER_SCOREBOARD_TASK_ID") or "").strip() or None


def run_eval(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> int:
    args.tasks = resolve_run_tasks(args, config)
    lighteval_plan = build_lighteval_plan(args, root=root, env=env, config=config)
    base_url = base_url_from_lighteval_command(lighteval_plan.command) or local_openai_base_url(
        config, env, args
    )

    manage_server = not getattr(args, "no_server", False) and is_local_base_url(base_url)
    infer_plan: CommandPlan | None = None
    if manage_server:
        infer_plan = build_infer_plan(
            infer_args_namespace(args, port=port_from_base_url(base_url)),
            root=root,
            env=env,
            config=config,
        )

    if args.dry_run:
        if infer_plan is not None:
            print(format_plan_for_display(infer_plan))
        print(format_plan_for_display(lighteval_plan))
        return 0

    api_key = lighteval_plan.env.get("HELICOPTER_EVAL_API_KEY") or lighteval_plan.env.get(
        "OPENAI_API_KEY"
    )
    allow_model_mismatch = lighteval_plan.env.get(
        "HELICOPTER_ALLOW_MODEL_MISMATCH", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    expected_model = getattr(args, "lighteval_model_name", None)
    # External endpoints are already running, so validate before creating a
    # scoreboard task. Managed endpoints are checked again after startup below.
    if not manage_server:
        validate_endpoint_model(
            base_url,
            expected_model,
            api_key=api_key,
            allow_mismatch=allow_model_mismatch,
        )
    database_only = lighteval_plan.env.get("HELICOPTER_SCOREBOARD_DB_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    scoreboard_task_id: str | None = None
    pinned_scoreboard_task_id = _pinned_scoreboard_task_id(lighteval_plan.env)
    if database_only:
        selected_tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
        if len(selected_tasks) != 1:
            raise SystemExit("database pipeline requires one (model, benchmark) task per process")

        dataset = scoreboard_dataset_name(selected_tasks[0])
        if pinned_scoreboard_task_id is not None:
            scoreboard_task_id = pinned_scoreboard_task_id
        else:
            from .scoreboard_bridge import prepare_lighteval_task

            with SCOREBOARD_LOCK, _scoreboard_env(lighteval_plan.env):
                scoreboard_task_id = prepare_lighteval_task(
                    model=scoreboard_model_name(args, config),
                    dataset=dataset,
                    root=root,
                    env=lighteval_plan.env,
                )
        lighteval_plan.env["HELICOPTER_SCOREBOARD_DATASET"] = dataset
        if scoreboard_task_id:
            lighteval_plan.env["HELICOPTER_SCOREBOARD_TASK_ID"] = scoreboard_task_id
        else:
            lighteval_plan.env.pop("HELICOPTER_SCOREBOARD_TASK_ID", None)
            if str(lighteval_plan.env.get("HELICOPTER_PIPELINE_STAGE") or "").lower() == "score":
                raise RuntimeError(
                    f"score stage has no persisted completions for {dataset!r}"
                )
        args.scoreboard_task_id = scoreboard_task_id or None

    output_dir = output_dir_from_command(lighteval_plan.command) or (root / "results/lighteval")
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if manage_server and infer_plan is not None:
        if server_is_healthy(base_url):
            print(f"eval run: reusing healthy server at {base_url}")
        else:
            server_log = output_dir / "server_logs" / f"vllm_{stamp}.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"eval run: starting vLLM server (log: {server_log})")
            with server_log.open("wb") as log_file:
                server_process = subprocess.Popen(
                    infer_plan.command,
                    cwd=str(infer_plan.cwd) if infer_plan.cwd else None,
                    env=infer_plan.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            timeout_s = float(
                pick(getattr(args, "server_timeout", None), DEFAULT_SERVER_TIMEOUT_S)
            )
            wait_for_server(base_url, process=server_process, log_path=server_log, timeout_s=timeout_s)
            print(f"eval run: server healthy at {base_url}")

    if manage_server:
        validate_endpoint_model(
            base_url,
            expected_model,
            api_key=api_key,
            allow_mismatch=allow_model_mismatch,
        )

    performance_output = output_dir / "performance" / f"performance_{stamp}.json"
    metrics_url = getattr(args, "metrics_url", None) or derive_metrics_url(base_url)
    try:
        exit_code = run_lighteval_with_performance(
            lighteval_plan.command,
            cwd=lighteval_plan.cwd,
            env=lighteval_plan.env,
            root=root,
            performance_output=performance_output,
            metrics_url=metrics_url,
            scoreboard_task_id=getattr(args, "scoreboard_task_id", None),
        )
    finally:
        if server_process is not None and not getattr(args, "keep_server", False):
            print("eval run: stopping vLLM server")
            stop_server(server_process)
        elif server_process is not None:
            print(f"eval run: leaving vLLM server running (pid {server_process.pid})")
    if scoreboard_task_id and exit_code != 0 and pinned_scoreboard_task_id is None:
        from .scoreboard_bridge import set_lighteval_task_status

        with SCOREBOARD_LOCK, _scoreboard_env(lighteval_plan.env):
            set_lighteval_task_status(task_id=scoreboard_task_id, status="Failed")

    database_only = lighteval_plan.env.get("HELICOPTER_SCOREBOARD_DB_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}
    if exit_code == 0 and getattr(args, "scoreboard", False) and not database_only:
        try:
            report = json.loads(performance_output.read_text())
        except (OSError, json.JSONDecodeError):
            report = {}
        result_files = report.get("source_files", {}).get("results", [])
        detail_files = report.get("source_files", {}).get("details", [])
        ingest_scoreboard_results(
            result_files=list(result_files),
            detail_files=list(detail_files),
            model_name=scoreboard_model_name(args, config),
            root=root,
            env=env,
        )
    suffix = "database-only mode" if database_only else f"performance report: {performance_output}"
    print(f"eval run: finished with exit code {exit_code}; {suffix}")
    return exit_code
