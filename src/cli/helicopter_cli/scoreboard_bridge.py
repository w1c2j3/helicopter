from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from helicopter_cli.lighteval_answer_adapters import adapt_answer


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


def _compact_lighteval_doc(value: Any) -> dict[str, Any]:
    """Keep only non-reconstructible LightEval document identity in completions.

    The exact prompt and model response are already stored in stages and
    model_response. Persisting the full dataset document for every rollout
    duplicates large code-test payloads (eight times with avg@8). Arena Hard
    additionally needs its reference answer while scoring from DB checkpoints,
    so retain that small field together with stable sample identity.
    """

    payload = _jsonable(value)
    if not isinstance(payload, Mapping):
        return {}
    compact = {
        key: payload[key]
        for key in ("id", "task_name")
        if payload.get(key) is not None
    }
    specific = payload.get("specific")
    if isinstance(specific, Mapping):
        kept = {
            key: specific[key]
            for key in ("sample_id", "references", "reference", "reference_answer", "reference_answers")
            if specific.get(key) is not None
        }
        if kept:
            compact["specific"] = kept
    return compact


def _response_payload(value: Any) -> dict[str, Any]:
    payload = _jsonable(value)
    if not isinstance(payload, dict):
        return {}
    # Raw endpoint metadata is attached dynamically to ModelResponse because
    # upstream LightEval has no finish_reason/usage fields. dataclasses.asdict
    # intentionally ignores those attributes, so preserve them explicitly.
    for key in (
        "raw_text",
        "finish_reason",
        "usage",
        "stages",
        "stages_by_rollout",
        "helicopter_rollout_scores",
    ):
        if hasattr(value, key):
            payload[key] = _jsonable(getattr(value, key))
    return payload


def _rollout_count(response: Mapping[str, Any]) -> int:
    sizes = [
        len(value)
        for key in ("text", "text_post_processed", "raw_text", "finish_reason", "stages_by_rollout")
        if isinstance((value := response.get(key)), list) and value
    ]
    return max(sizes, default=1)


def _rollout_usage(value: Any, index: int) -> Any:
    """Select one rollout from usage metadata, including legacy nested copies."""

    if isinstance(value, list):
        if not value:
            return None
        selected = value[index] if index < len(value) else value[0]
        return _rollout_usage(selected, index)
    if isinstance(value, Mapping):
        return {str(key): _rollout_usage(item, index) for key, item in value.items()}
    return value


def _rollout_response(response: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = dict(response)
    for key in ("text", "text_post_processed", "raw_text", "finish_reason"):
        value = response.get(key)
        if isinstance(value, list):
            result[key] = value[index] if index < len(value) else None
    stages = response.get("stages_by_rollout")
    if isinstance(stages, list):
        result["stages"] = stages[index] if index < len(stages) else None
    if "usage" in response:
        result["usage"] = _rollout_usage(response.get("usage"), index)
    result.pop("stages_by_rollout", None)
    return result


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
    for key in ("text_post_processed", "text"):
        value = response.get(key)
        if value is not None:
            return _text(value)
    return json.dumps(_jsonable(response), ensure_ascii=False, sort_keys=True)


def _completion_answer(response: Mapping[str, Any]) -> str:
    answer = _text(response.get("text"))
    if answer:
        return answer
    logprobs = response.get("logprobs")
    if isinstance(logprobs, list) and logprobs:
        try:
            choice_index = max(range(len(logprobs)), key=lambda index: float(logprobs[index]))
        except (TypeError, ValueError):
            choice_index = -1
        if choice_index >= 0:
            return f" {chr(ord('A') + choice_index) if choice_index < 26 else choice_index + 1}"
    return _answer(response)


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
        selected = []
        for index in indices:
            if not isinstance(index, int) or not 0 <= index < len(choices):
                continue
            value = _text(choices[index])
            # A single gold among multiple choices is an MCQ. Multiple gold
            # indices represent acceptable free-form aliases, not choices the
            # candidate must enumerate.
            if len(choices) > 1 and len(indices) == 1:
                label = chr(ord("A") + index) if index < 26 else str(index + 1)
                # The model-side choice adapter returns this same canonical
                # label. Keep the DB reference symmetric instead of storing
                # the full option text on only the gold side.
                value = f" {label}"
            selected.append(value)
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

def _rollout_official_result(
    response: Mapping[str, Any],
    repeat_index: int,
) -> tuple[bool | None, dict[str, float]]:
    recorded = response.get("helicopter_rollout_scores")
    if not isinstance(recorded, Mapping):
        return None, {}
    scores: dict[str, float] = {}
    for name, values in recorded.items():
        if not isinstance(values, list) or repeat_index >= len(values):
            continue
        try:
            scores[str(name)] = float(values[repeat_index])
        except (TypeError, ValueError):
            continue
    if not scores:
        return None, {}
    primary = next(iter(scores.values()))
    return primary > 0.0, scores



def _json_env(name: str, values: Mapping[str, str] | None = None) -> dict[str, Any]:
    source = os.environ if values is None else values
    raw = source.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def sampling_config_from_env(values: Mapping[str, str]) -> dict[str, Any]:
    sampling = _json_env("HELICOPTER_VLLM_SAMPLING_JSON", values)
    policy = _json_env("HELICOPTER_LIGHTEEVAL_G1H_POLICY", values)
    for key in (
        "metric",
        "avg_k",
        "rollout_n",
        "generation_size",
        "gpass_generation_size",
        "zero_shot",
        "prompt_style",
    ):
        if key in policy:
            sampling[key] = policy[key]
    prompt_mode = values.get("HELICOPTER_SCOREBOARD_PROMPT_MODE", "").strip()
    if prompt_mode:
        sampling["prompt_mode"] = prompt_mode
    request_policy = _json_env("HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY", values)
    task_policies = request_policy.get("tasks")
    if isinstance(task_policies, Mapping) and len(task_policies) == 1:
        task_name, task_policy = next(iter(task_policies.items()))
        if isinstance(task_policy, Mapping):
            task_request_policy = {
                "task": str(task_name),
                "domain": task_policy.get("domain"),
                "format": task_policy.get("format"),
                "prompt_template": task_policy.get("prompt_template"),
                "inherit_task_stops": bool(task_policy.get("inherit_task_stops", True)),
                "stop": _jsonable(task_policy.get("stop")),
                "sampling": _jsonable(task_policy.get("sampling")),
            }
            multi_turn_template = task_policy.get("multi_turn_template")
            if multi_turn_template is not None:
                task_request_policy["multi_turn_template"] = multi_turn_template
            sampling["task_request_policy"] = task_request_policy
    return sampling


def _sampling_config() -> dict[str, Any]:
    return sampling_config_from_env(os.environ)



def _dataset(task: str) -> str:
    parts = [p.strip() for p in str(task).split("|") if p.strip()]
    if len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    name = parts[-1] if parts else str(task)
    return name[len("g1h__") :] if name.startswith("g1h__") else name


def _add_scoreboard(root: Path) -> None:
    path = root / "src/scoreboard-server"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def _db_env(values: Mapping[str, str] | None):
    changed: dict[str, str | None] = {}
    for key, value in (values or {}).items():
        if key.startswith(("SCOREBOARD_DB_", "PG", "HELICOPTER_")) and os.environ.get(key) != str(value):
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


async def _write(
    model: str,
    job: str,
    rows: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    mark_completed: bool = True,
    pinned_task_id_override: str | None = None,
) -> list[str]:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    recorded: list[str] = []
    cot_mode = os.environ.get("HELICOPTER_SCOREBOARD_COT_MODE", "CoT")
    sampling = _sampling_config()
    config_path = os.environ.get("HELICOPTER_SCOREBOARD_CONFIG_PATH")
    pinned_task_id = (
        pinned_task_id_override or os.environ.get("HELICOPTER_SCOREBOARD_TASK_ID", "").strip()
    )
    try:
        store = ScoreboardStore(settings=settings)
        for task, (completions, evals) in rows.items():
            dataset = _dataset(task)
            sample_count = len({int(item["sample_index"]) for item in completions})
            requested_task_id = pinned_task_id if pinned_task_id and len(rows) == 1 else None
            task_id, _inserted = await store.insert_completion_payloads_with_task(
                payloads=completions,
                task_id=requested_task_id,
                job_name=job,
                dataset=dataset,
                model=model,
                is_param_search=False,
                sampling_config=sampling,
                config_path=config_path,
                allow_resume=True,
                num_samples=sample_count,
            )
            if task_id is None:
                continue
            if not mark_completed:
                await store.update_task_status(task_id=task_id, status="Running")
            if mark_completed:
                await store.ingest_eval_payloads(payloads=evals, task_id=task_id)
                await store.record_score_payload(
                    task_id=task_id,
                    payload={"cot_mode": cot_mode, "metrics": _jsonable(metrics.get(task, {}))},
                    mark_completed=True,
                )
            recorded.append(f"{dataset} -> task {task_id}")
    finally:
        await close_db()
    return recorded


def write_lighteval_tracker(tracker: Any) -> list[str]:
    root = Path(os.environ.get("HELICOPTER_PROJECT_ROOT") or Path.cwd()).resolve()
    _add_scoreboard(root)
    model = str(os.environ.get("HELICOPTER_SCOREBOARD_MODEL_NAME") or tracker.general_config_logger.model_name)
    rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    sampling = _sampling_config()
    request_policy = _json_env("HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY")
    configured_task_policy: Mapping[str, Any] = {}
    task_policies = request_policy.get("tasks")
    if isinstance(task_policies, Mapping) and len(task_policies) == 1:
        only_policy = next(iter(task_policies.values()))
        if isinstance(only_policy, Mapping):
            configured_task_policy = only_policy
    for task, details in tracker.details_logger.details.items():
        completions: list[dict[str, Any]] = []
        evals: list[dict[str, Any]] = []
        for index, detail in enumerate(details):
            doc = _jsonable(detail.doc)
            response = _response_payload(detail.model_response)
            metric = _jsonable(detail.metric)
            prompt = response.get("input") or doc.get("query") or ""
            for repeat_index in range(_rollout_count(response)):
                rollout = _rollout_response(response, repeat_index)
                completion_answer = _completion_answer(rollout)
                answer = adapt_answer(
                    completion_answer,
                    domain=configured_task_policy.get("domain"),
                    request_format=configured_task_policy.get("format"),
                    prompt=str(prompt),
                    stops=(
                        configured_task_policy.get("stop")
                        if isinstance(configured_task_policy.get("stop"), list)
                        else []
                    ),
                )
                official_passed, rollout_metrics = _rollout_official_result(response, repeat_index)
                passed = _passed(metric) if official_passed is None else official_passed
                key = {"sample_index": index, "repeat_index": repeat_index, "pass_index": 0}
                stage_payload = _completion_stages(
                    rollout,
                    fallback_prompt=prompt,
                    fallback_completion=completion_answer,
                )
                completions.append(
                    {
                        **key,
                        **stage_payload,
                        "sampling_config": sampling,
                        "stats": {
                            "metrics": metric,
                            "rollout_metrics": rollout_metrics,
                            "lighteval_task": task,
                        },
                        "agent_result": {
                            "doc": _compact_lighteval_doc(doc),
                            "model_response": rollout,
                        },
                        "task_id": doc.get("id"),
                    }
                )
                reference = adapt_answer(
                    _reference(doc),
                    domain=configured_task_policy.get("domain"),
                    request_format=configured_task_policy.get("format"),
                    prompt=str(prompt),
                    stops=(
                        configured_task_policy.get("stop")
                        if isinstance(configured_task_policy.get("stop"), list)
                        else []
                    ),
                )
                evals.append(
                    {
                        **key,
                        "answer": answer,
                        "ref_answer": reference,
                        "raw_record": doc,
                        "is_passed": passed,
                        "fail_reason": ""
                        if passed
                        else json.dumps(metric, ensure_ascii=False, sort_keys=True),
                    }
                )
        rows[str(task)] = (completions, evals)
    official_metrics = {
        str(task): _jsonable(value)
        for task, value in tracker.metrics_logger.metric_aggregated.items()
    }
    with _LOCK:
        return asyncio.run(_write(model, "lighteval", rows, official_metrics))


def write_function_calling_results(
    *,
    samples: Iterable[Any],
    results: Iterable[Any],
    metrics: Mapping[str, Mapping[str, Any]],
    model: str,
    root: Path,
    env: Mapping[str, str],
    task_id: str | None = None,
) -> list[str]:
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
        return asyncio.run(
            _write(model, "function_calling", rows, metrics, pinned_task_id_override=task_id)
        )

async def _prepare_function_calling_task(
    *,
    model: str,
    dataset: str,
    sampling_config: Mapping[str, Any],
    config_path: str | None,
) -> str:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        ctx = await store.get_resume_context(
            job_name="function_calling",
            dataset=dataset,
            model=model,
            is_param_search=False,
            sampling_config=_jsonable(sampling_config),
            config_path=config_path,
        )
        if not ctx.can_resume or ctx.task_id is None:
            return ""
        task_id = str(ctx.task_id)
        if await store.count_completions(task_id=task_id, status="Completed") <= 0:
            return ""
        await store.update_task_status(task_id=task_id, status="Running")
        return task_id
    finally:
        await close_db()


def prepare_function_calling_task(
    *,
    model: str,
    dataset: str,
    sampling_config: Mapping[str, Any],
    config_path: str | None,
    root: Path,
    env: Mapping[str, str],
) -> str:
    """Find an FC task that already owns real generation checkpoints.

    A new task is not created here. The first real function-calling result
    creates its task and completion together in the checkpoint transaction.
    """

    _add_scoreboard(root)
    with _LOCK, _db_env(env):
        return asyncio.run(
            _prepare_function_calling_task(
                model=model,
                dataset=dataset,
                sampling_config=sampling_config,
                config_path=config_path,
            )
        )


async def _checkpoint_function_calling_result(
    *,
    task_id: str | None,
    dataset: str,
    model: str,
    num_samples: int,
    payload: dict[str, Any],
    sampling_config: Mapping[str, Any],
    config_path: str | None,
) -> tuple[str, int]:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        resolved_task_id, inserted = await store.insert_completion_payloads_with_task(
            payloads=[payload],
            task_id=task_id,
            job_name="function_calling",
            dataset=dataset,
            model=model,
            is_param_search=False,
            sampling_config=_jsonable(sampling_config),
            config_path=config_path,
            allow_resume=True,
            num_samples=num_samples,
        )
        if resolved_task_id is None:
            raise RuntimeError("function-calling checkpoint did not persist a task")
        return str(resolved_task_id), inserted
    finally:
        await close_db()


def checkpoint_function_calling_result(
    *,
    task_id: str | None,
    dataset: str,
    model: str,
    num_samples: int,
    sample_index: int,
    sample: Any,
    result: Any,
    sampling_config: Mapping[str, Any],
    config_path: str | None,
    root: Path,
    env: Mapping[str, str],
) -> str:
    prompt = json.dumps(
        {"messages": _jsonable(sample.messages), "tools": _jsonable(sample.tools)},
        ensure_ascii=False,
        sort_keys=True,
    )
    raw_response = _jsonable(result.raw_response)
    completion = json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    payload = {
        "_stage": "generation",
        "status": "Completed" if result.error is None else "Failed",
        "sample_index": int(sample_index),
        "repeat_index": 0,
        "pass_index": 0,
        "prompt1": prompt,
        "completion1": completion,
        "stop_reason1": None,
        "sampling_config": _jsonable(sampling_config),
        "stats": {
            "generation_checkpoint": True,
            "error": result.error,
            "elapsed_seconds": result.elapsed_seconds,
        },
        "agent_result": {
            "sample": _jsonable(sample),
            "run_result": _jsonable(result),
        },
        "task_id": sample.sample_id,
    }
    _add_scoreboard(root)
    with _LOCK, _db_env(env):
        resolved_task_id, _inserted = asyncio.run(
            _checkpoint_function_calling_result(
                task_id=task_id,
                dataset=dataset,
                model=model,
                num_samples=num_samples,
                payload=payload,
                sampling_config=sampling_config,
                config_path=config_path,
            )
        )
    return resolved_task_id


def load_function_calling_generation(
    *,
    task_id: str,
    root: Path,
    env: Mapping[str, str],
) -> list[dict[str, Any]]:
    _add_scoreboard(root)
    with _LOCK, _db_env(env):
        return asyncio.run(_load_generation(task_id))


async def _prepare_lighteval_task(
    *,
    model: str,
    dataset: str,
) -> str:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        ctx = await store.get_resume_context(
            job_name="lighteval",
            dataset=dataset,
            model=model,
            is_param_search=False,
            sampling_config=_sampling_config(),
            config_path=os.environ.get("HELICOPTER_SCOREBOARD_CONFIG_PATH"),
        )
        if not ctx.can_resume or ctx.task_id is None:
            return ""
        task_id = str(ctx.task_id)
        if await store.count_completions(task_id=task_id, status="Completed") <= 0:
            return ""
        await store.update_task_status(task_id=task_id, status="Running")
        return task_id
    finally:
        await close_db()


def prepare_lighteval_task(
    *,
    model: str,
    dataset: str,
    root: Path,
    env: Mapping[str, str] | None = None,
) -> str:
    """Find a task that already owns real generation checkpoints.

    A new task is deliberately not created here. The first completed model
    response creates its task and completion together in the checkpoint
    transaction, so a request that fails before producing data leaves no
    result-table placeholder.
    """

    _add_scoreboard(root)
    with _LOCK, _db_env(env):
        return asyncio.run(_prepare_lighteval_task(model=model, dataset=dataset))


def _generation_payloads(
    *,
    task_name: str,
    sample_index: int,
    doc: Any,
    response: Any,
    repeat_indices: Iterable[int],
    generation_size: int | None = None,
) -> list[dict[str, Any]]:
    doc_payload = _jsonable(doc)
    if not isinstance(doc_payload, dict):
        doc_payload = {"value": doc_payload}
    response_payload = _response_payload(response)
    prompt = response_payload.get("input") or doc_payload.get("query") or ""
    indices = list(repeat_indices)
    payloads: list[dict[str, Any]] = []
    sampling_config = _sampling_config()
    if generation_size is not None:
        sampling_config["effective_generation_size"] = int(generation_size)
    for offset, repeat_index in enumerate(indices):
        rollout = _rollout_response(response_payload, offset)
        completion = _completion_answer(rollout)
        payloads.append(
            {
                "_stage": "generation",
                "status": "Completed",
                "sample_index": int(sample_index),
                "repeat_index": int(repeat_index),
                "pass_index": 0,
                **_completion_stages(
                    rollout,
                    fallback_prompt=str(prompt),
                    fallback_completion=completion,
                ),
                "sampling_config": sampling_config,
                "stats": {"lighteval_task": task_name, "generation_checkpoint": True},
                "agent_result": {
                    "doc": _compact_lighteval_doc(doc_payload),
                    "model_response": rollout,
                },
                "task_id": doc_payload.get("id"),
            }
        )
    return payloads


def _loglikelihood_payload(
    *,
    task_name: str,
    sample_index: int,
    doc: Any,
    response: Any,
) -> dict[str, Any]:
    """Build one durable completion row for a loglikelihood document."""

    doc_payload = _jsonable(doc)
    if not isinstance(doc_payload, dict):
        doc_payload = {"value": doc_payload}
    response_payload = _response_payload(response)
    prompt = _text(response_payload.get("input") or doc_payload.get("query") or "")
    return {
        "_stage": "generation",
        "status": "Completed",
        "sample_index": int(sample_index),
        "repeat_index": 0,
        "pass_index": 0,
        "prompt1": prompt,
        "completion1": _completion_answer(response_payload),
        "stop_reason1": None,
        "sampling_config": _sampling_config(),
        "stats": {"lighteval_task": task_name, "loglikelihood_checkpoint": True},
        "agent_result": {
            "doc": _compact_lighteval_doc(doc_payload),
            "model_response": response_payload,
        },
        "task_id": doc_payload.get("id"),
    }




async def _checkpoint_generation(
    *,
    task_id: str,
    dataset: str,
    num_samples: int,
    payloads: list[dict[str, Any]],
) -> int:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        await store.ensure_benchmark_num_samples(dataset=dataset, num_samples=num_samples)
        return await store.insert_completion_payloads_batch(payloads=payloads, task_id=task_id)
    finally:
        await close_db()


class LightevalCheckpointSession:
    """Persist generation checkpoints on one event loop and DB connection."""

    def __init__(
        self,
        *,
        task_id: str | None,
        dataset: str,
        num_samples: int,
        model: str | None = None,
    ) -> None:
        self.task_id = str(task_id or "").strip() or None
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.model = str(
            model or os.environ.get("HELICOPTER_SCOREBOARD_MODEL_NAME") or ""
        ).strip()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._store: Any | None = None

    async def _open(self) -> None:
        from scoreboard_server.db.connection import init_db
        from scoreboard_server.db.repository import ScoreboardStore
        from scoreboard_server.db.settings import DatabaseSettings

        settings = DatabaseSettings.from_env()
        await init_db(settings, generate_schemas=False)
        self._store = ScoreboardStore(settings=settings)

    async def _write(self, payloads: list[dict[str, Any]]) -> int:
        if self._store is None:
            raise RuntimeError("LightEval checkpoint session is not open")
        if self.task_id is None and not self.model:
            raise RuntimeError(
                "first LightEval checkpoint requires HELICOPTER_SCOREBOARD_MODEL_NAME"
            )

        is_first_write = self.task_id is None
        task_id, inserted = await self._store.insert_completion_payloads_with_task(
            payloads=payloads,
            task_id=self.task_id,
            job_name="lighteval",
            dataset=self.dataset,
            model=self.model,
            is_param_search=False,
            sampling_config=_sampling_config(),
            config_path=os.environ.get("HELICOPTER_SCOREBOARD_CONFIG_PATH"),
            allow_resume=True,
            num_samples=self.num_samples,
        )
        if task_id is None:
            return inserted
        self.task_id = str(task_id)
        if is_first_write:
            os.environ["HELICOPTER_SCOREBOARD_TASK_ID"] = self.task_id
        return inserted

    async def _close(self) -> None:
        from scoreboard_server.db.connection import close_db

        try:
            await close_db()
        finally:
            self._store = None

    def __enter__(self) -> LightevalCheckpointSession:
        if self._loop is not None:
            raise RuntimeError("LightEval checkpoint session is already open")
        root = Path(os.environ.get("HELICOPTER_PROJECT_ROOT") or Path.cwd()).resolve()
        _add_scoreboard(root)
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._open())
        except BaseException:
            loop.close()
            self._loop = None
            raise
        return self

    def checkpoint(
        self,
        *,
        task_name: str,
        sample_index: int,
        doc: Any,
        response: Any,
        repeat_indices: Iterable[int],
        generation_size: int | None = None,
    ) -> int:
        if self._loop is None:
            raise RuntimeError("LightEval checkpoint session is not open")
        payloads = _generation_payloads(
            task_name=task_name,
            sample_index=sample_index,
            doc=doc,
            response=response,
            repeat_indices=repeat_indices,
            generation_size=generation_size,
        )
        inserted = 0
        # Keep each completed rollout durable independently.  A request with
        # n>1 returns several choices at once, but a later process failure
        # must not erase the choices that were already available.
        for payload in payloads:
            inserted += self._loop.run_until_complete(self._write([payload]))
        return inserted

    def checkpoint_loglikelihood(
        self,
        *,
        task_name: str,
        sample_index: int,
        doc: Any,
        response: Any,
    ) -> int:
        if self._loop is None:
            raise RuntimeError("LightEval checkpoint session is not open")
        payload = _loglikelihood_payload(
            task_name=task_name,
            sample_index=sample_index,
            doc=doc,
            response=response,
        )
        return self._loop.run_until_complete(self._write([payload]))


    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.run_until_complete(self._close())
        finally:
            loop.close()
            self._loop = None


def checkpoint_lighteval_response(
    *,
    task_id: str,
    dataset: str,
    task_name: str,
    num_samples: int,
    sample_index: int,
    doc: Any,
    response: Any,
    repeat_indices: Iterable[int],
    generation_size: int | None = None,
) -> int:
    """Persist one prompt's finished rollouts before the next model work."""
    with _LOCK:
        with LightevalCheckpointSession(
            task_id=task_id,
            dataset=dataset,
            num_samples=num_samples,
        ) as session:
            return session.checkpoint(
                task_name=task_name,
                sample_index=sample_index,
                doc=doc,
                response=response,
                repeat_indices=repeat_indices,
                generation_size=generation_size,
            )


async def _load_generation(task_id: str) -> list[dict[str, Any]]:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        return await store.list_completion_payloads(task_id=task_id)
    finally:
        await close_db()


def load_lighteval_generation(*, task_id: str) -> list[dict[str, Any]]:
    root = Path(os.environ.get("HELICOPTER_PROJECT_ROOT") or Path.cwd()).resolve()
    _add_scoreboard(root)
    with _LOCK:
        return asyncio.run(_load_generation(task_id))


async def _set_task_status(task_id: str, status: str) -> None:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        await ScoreboardStore(settings=settings).update_task_status(task_id=task_id, status=status)
    finally:
        await close_db()


def set_function_calling_task_status(
    *,
    task_id: str,
    status: str,
    root: Path,
    env: Mapping[str, str],
) -> None:
    """Update a function-calling task using its explicit database environment."""

    _add_scoreboard(root)
    with _LOCK, _db_env(env):
        asyncio.run(_set_task_status(task_id, status))


def set_lighteval_task_status(*, task_id: str, status: str) -> None:
    root = Path(os.environ.get("HELICOPTER_PROJECT_ROOT") or Path.cwd()).resolve()
    _add_scoreboard(root)
    with _LOCK:
        asyncio.run(_set_task_status(task_id, status))
