from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _response_payload(value: Any) -> dict[str, Any]:
    payload = _jsonable(value)
    if not isinstance(payload, dict):
        return {}
    # Raw endpoint metadata is attached dynamically to ModelResponse because
    # upstream LightEval has no finish_reason/usage fields. dataclasses.asdict
    # intentionally ignores those attributes, so preserve them explicitly.
    for key in ("raw_text", "finish_reason", "usage", "stages"):
        if hasattr(value, key):
            payload[key] = _jsonable(getattr(value, key))
    return payload


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        values = [v for v in value if v not in (None, "")]
        if len(values) == 1:
            return str(values[0])
        if values:
            return json.dumps(_jsonable(values), ensure_ascii=False, sort_keys=True)
    return "" if value in (None, "") else str(value)


def _answer(response: Mapping[str, Any]) -> str:
    return next((text for key in ("text_post_processed", "text") if (text := _text(response.get(key)))), json.dumps(_jsonable(response), ensure_ascii=False, sort_keys=True))


def _completion_answer(response: Mapping[str, Any]) -> str:
    return _text(response.get("text")) or _answer(response)


def _stop_reason(response: Mapping[str, Any]) -> str | None:
    value = response.get("finish_reason")
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else value
    if value in (None, "", []):
        return None
    return _text(value)


def _completion_stages(
    response: Mapping[str, Any], *, fallback_prompt: str, fallback_completion: str
) -> dict[str, Any]:
    stages = response.get("stages")
    if not isinstance(stages, list) or not stages:
        return {
            "prompt1": fallback_prompt,
            "completion1": fallback_completion,
            "stop_reason1": _stop_reason(response),
        }
    payload: dict[str, Any] = {}
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            continue
        payload[f"prompt{index}"] = _text(stage.get("prompt"))
        payload[f"completion{index}"] = _text(stage.get("completion"))
        stop_reason = stage.get("stop_reason")
        if isinstance(stop_reason, list):
            stop_reason = stop_reason[0] if len(stop_reason) == 1 else stop_reason
        payload[f"stop_reason{index}"] = None if stop_reason in (None, "", []) else _text(stop_reason)
    return payload or {
        "prompt1": fallback_prompt,
        "completion1": fallback_completion,
        "stop_reason1": _stop_reason(response),
    }


def _reference(doc: Mapping[str, Any]) -> str:
    choices, indices = doc.get("choices"), doc.get("gold_index")
    if isinstance(indices, int):
        indices = [indices]
    if isinstance(choices, list) and isinstance(indices, list):
        selected = [choices[i] for i in indices if isinstance(i, int) and 0 <= i < len(choices)]
        if selected:
            return _text(selected)
    for source in (doc, doc.get("specific")):
        if isinstance(source, Mapping):
            for key in ("expected_answer", "reference_answer", "solution", "answer", "target"):
                if source.get(key) not in (None, ""):
                    return _text(source[key])
    return ""


def _passed(metrics: Mapping[str, Any]) -> bool:
    for name, value in metrics.items():
        if "stderr" in str(name).lower():
            continue
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _dataset(task: str) -> str:
    parts = [p.strip() for p in str(task).split("|") if p.strip()]
    if len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    return parts[-1] if parts else str(task)


def _add_scoreboard(root: Path) -> None:
    path = root / "src/scoreboard-server"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def _db_env(values: Mapping[str, str] | None):
    changed: dict[str, str | None] = {}
    for key, value in (values or {}).items():
        if key.startswith(("SCOREBOARD_DB_", "PG")) and os.environ.get(key) != str(value):
            changed[key] = os.environ.get(key)
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, previous in changed.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


async def _write(model: str, job: str, rows: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]], metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    recorded: list[str] = []
    cot_mode = os.environ.get("HELICOPTER_SCOREBOARD_COT_MODE", "CoT")
    try:
        store = ScoreboardStore(settings=settings)
        for task, (completions, evals) in rows.items():
            dataset = _dataset(task)
            task_id = await store.get_or_create_task(job_name=job, job_id=None, dataset=dataset, model=model, is_param_search=False, allow_resume=False)
            await store.ensure_benchmark_num_samples(dataset=dataset, num_samples=len(completions))
            await store.insert_completion_payloads_batch(payloads=completions, task_id=task_id)
            await store.ingest_eval_payloads(payloads=evals, task_id=task_id)
            await store.record_score_payload(task_id=task_id, payload={"cot_mode": cot_mode, "metrics": _jsonable(metrics.get(task, {}))})
            recorded.append(f"{dataset} -> task {task_id}")
    finally:
        await close_db()
    return recorded


def write_lighteval_tracker(tracker: Any) -> list[str]:
    root = Path(os.environ.get("HELICOPTER_PROJECT_ROOT") or Path.cwd()).resolve()
    _add_scoreboard(root)
    model = str(os.environ.get("HELICOPTER_SCOREBOARD_MODEL_NAME") or tracker.general_config_logger.model_name)
    rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for task, details in tracker.details_logger.details.items():
        completions: list[dict[str, Any]] = []
        evals: list[dict[str, Any]] = []
        for index, detail in enumerate(details):
            doc = _jsonable(detail.doc)
            response = _response_payload(detail.model_response)
            metric = _jsonable(detail.metric)
            completion_answer, answer, passed = _completion_answer(response), _answer(response), _passed(metric)
            key = {"sample_index": index, "repeat_index": 0, "pass_index": 0}
            prompt = response.get("input") or doc.get("query") or ""
            stage_payload = _completion_stages(response, fallback_prompt=prompt, fallback_completion=completion_answer)
            completions.append({**key, **stage_payload, "stats": {"metrics": metric, "lighteval_task": task}, "agent_result": {"doc": doc, "model_response": response}, "task_id": doc.get("id")})
            evals.append({**key, "answer": answer, "ref_answer": _reference(doc), "raw_record": doc, "is_passed": passed, "fail_reason": "" if passed else json.dumps(metric, ensure_ascii=False, sort_keys=True)})
        rows[str(task)] = (completions, evals)
    metrics = {str(task): _jsonable(value) for task, value in tracker.metrics_logger.metric_aggregated.items()}
    with _LOCK:
        return asyncio.run(_write(model, "lighteval", rows, metrics))


def write_function_calling_results(*, samples: Iterable[Any], results: Iterable[Any], metrics: Mapping[str, Mapping[str, Any]], model: str, root: Path, env: Mapping[str, str]) -> list[str]:
    _add_scoreboard(root)
    sample_map = {(s.task_name, s.sample_id): s for s in samples}
    grouped: dict[str, list[Any]] = defaultdict(list)
    for result in results:
        grouped[result.task_name].append(result)
    rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for task, task_results in grouped.items():
        completions: list[dict[str, Any]] = []
        evals: list[dict[str, Any]] = []
        for index, result in enumerate(task_results):
            sample = sample_map[(result.task_name, result.sample_id)]
            key = {"sample_index": index, "repeat_index": 0, "pass_index": 0}
            answer = json.dumps(_jsonable(result.actual_calls), ensure_ascii=False, sort_keys=True)
            reference = json.dumps(_jsonable(sample.specific), ensure_ascii=False, sort_keys=True)
            passed = float(result.score) >= 1 and not result.error
            prompt = json.dumps({"messages": _jsonable(sample.messages), "tools": _jsonable(sample.tools)}, ensure_ascii=False, sort_keys=True)
            stats = {"score": result.score, "elapsed_seconds": result.elapsed_seconds, "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "total_tokens": result.total_tokens, "error": result.error}
            completions.append({**key, "prompt1": prompt, "completion1": answer, "stop_reason1": None, "stats": stats, "agent_result": _jsonable(result.raw_response), "task_id": result.sample_id})
            evals.append({**key, "answer": answer, "ref_answer": reference, "raw_record": _jsonable(sample.specific), "is_passed": passed, "fail_reason": "" if passed else str(result.error or f"score={result.score}")})
        rows[task] = (completions, evals)
    with _LOCK, _db_env(env):
        return asyncio.run(_write(model, "function_calling", rows, metrics))
