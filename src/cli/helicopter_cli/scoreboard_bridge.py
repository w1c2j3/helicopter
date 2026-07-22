from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

_LOCK = threading.Lock()
_JUDGE_CONTRACT = "answer-only-knowledge-v2"


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
            for key in ("sample_id", "references")
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
                value = f"{label}. {value}"
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



def _json_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def _sampling_config() -> dict[str, Any]:
    sampling = _json_env("HELICOPTER_VLLM_SAMPLING_JSON")
    policy = _json_env("HELICOPTER_LIGHTEEVAL_G1H_POLICY")
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
    prompt_mode = os.environ.get("HELICOPTER_SCOREBOARD_PROMPT_MODE", "").strip()
    if prompt_mode:
        sampling["prompt_mode"] = prompt_mode
    request_policy = _json_env("HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY")
    task_policies = request_policy.get("tasks")
    if isinstance(task_policies, Mapping) and len(task_policies) == 1:
        task_name, task_policy = next(iter(task_policies.items()))
        if isinstance(task_policy, Mapping):
            sampling["task_request_policy"] = {
                "task": str(task_name),
                "domain": task_policy.get("domain"),
                "prompt_template": task_policy.get("prompt_template"),
                "inherit_task_stops": bool(task_policy.get("inherit_task_stops", True)),
                "stop": _jsonable(task_policy.get("stop")),
                "sampling": _jsonable(task_policy.get("sampling")),
            }
    return sampling


def _judge_settings() -> tuple[str, str, str] | None:
    base_url = os.environ.get("HELICOPTER_JUDGE_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("HELICOPTER_JUDGE_API_KEY", "").strip()
    model = os.environ.get("HELICOPTER_JUDGE_MODEL", "").strip()
    if not any((base_url, api_key, model)):
        return None
    if not all((base_url, api_key, model)):
        raise RuntimeError(
            "judge requires HELICOPTER_JUDGE_BASE_URL, HELICOPTER_JUDGE_API_KEY, "
            "and HELICOPTER_JUDGE_MODEL"
        )
    url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
    return url, api_key, model


def _judge_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        value = "\n".join(lines).strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"judge returned invalid JSON: {value[:240]!r}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool):
        raise RuntimeError(f"judge response must contain boolean passed: {payload!r}")
    return {"passed": payload["passed"], "reason": str(payload.get("reason") or "").strip()}


def _judge_reference_payload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    match = re.match(r"^\s*([A-Z])\.\s+\S", value)
    return [match.group(1), value] if match else value


def _judge_one(item: Mapping[str, Any], settings: tuple[str, str, str]) -> dict[str, Any]:
    url, api_key, model = settings
    content = json.dumps(
        {
            "reference_answer": _judge_reference_payload(item["reference_answer"]),
            "candidate_answer": item["candidate_answer"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Judge whether candidate_answer gives the answer represented by reference_answer. "
                    "Use only those two user fields; exact wording is not required. A reference JSON "
                    "array contains alternative acceptable answers, so matching any one item is enough. "
                    "Accept mathematically equivalent expressions, semantic equivalence, aliases, and "
                    "harmless formatting differences. Pass an explanatory sentence when it clearly "
                    "identifies exactly one reference answer and adds no conflicting answer or false "
                    "claim. Reject answers that merely mention the reference while asserting something "
                    "different, offer incompatible alternatives, contradict it, omit material answer "
                    "content, or add a false claim. Never infer from a hidden question or outside "
                    "knowledge. Return only JSON: "
                    "{\"passed\": true|false, \"reason\": \"brief reason\"}."
                ),
            },
            {"role": "user", "content": content},
        ],
    }
    try:
        timeout = max(1, int(os.environ.get("HELICOPTER_JUDGE_TIMEOUT", "120")))
        retries = max(1, int(os.environ.get("HELICOPTER_JUDGE_MAX_RETRIES", "2")))
    except ValueError as error:
        raise RuntimeError("judge timeout/retry settings must be integers") from error
    last_error: Exception | None = None
    for _attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError(f"judge response has no choices: {body!r}")
            result = _judge_json(choices[0].get("message", {}).get("content", ""))
            score = float(result["passed"])
            return {
                **result,
                "score": score,
                "model": model,
                "contract": _JUDGE_CONTRACT,
            }
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(f"judge request failed after {retries} attempts: {last_error}")


def _completion_key(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(payload.get("sample_index", 0)),
        int(payload.get("repeat_index", payload.get("avg_repeat_index", 0))),
        int(payload.get("pass_index", 0)),
    )


def _stored_judge_result(completion: Mapping[str, Any]) -> dict[str, Any] | None:
    stats = completion.get("stats")
    result = stats.get("judge") if isinstance(stats, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or not isinstance(result.get("passed"), bool)
        or not isinstance(result.get("score"), (int, float))
        or result.get("contract") != _JUDGE_CONTRACT
    ):
        return None
    return dict(result)


def _record_judge_result(
    completion: dict[str, Any],
    evaluation: dict[str, Any],
    result: Mapping[str, Any],
) -> float:
    score = float(result["score"])
    passed = score >= 1.0
    completion.setdefault("stats", {})["judge"] = dict(result)
    evaluation["is_passed"] = passed
    evaluation["fail_reason"] = (
        "" if passed else str(result.get("reason") or "judge rejected")
    )
    return score


async def _restore_judge_checkpoints(
    task_id: str,
    completions: list[dict[str, Any]],
    evals: list[dict[str, Any]],
) -> int:
    from scoreboard_server.db.models.completion import Completion

    completion_targets = {_completion_key(item): item for item in completions}
    eval_targets = {_completion_key(item): item for item in evals}
    restored = 0
    rows = await Completion.filter(task_id=int(task_id)).only(
        "sample_index",
        "avg_repeat_index",
        "pass_index",
        "context",
    )
    for row in rows:
        key = (row.sample_index, row.avg_repeat_index, row.pass_index)
        completion = completion_targets.get(key)
        evaluation = eval_targets.get(key)
        if completion is None or evaluation is None or _stored_judge_result(completion) is not None:
            continue
        context = row.context if isinstance(row.context, Mapping) else {}
        result = _stored_judge_result(context)
        if result is None:
            continue
        _record_judge_result(completion, evaluation, result)
        restored += 1
    return restored


async def _persist_judge_checkpoint_batch_async(
    task_id: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings
    from tortoise.transactions import in_transaction

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        async with in_transaction():
            await store.insert_completion_payloads_batch(
                payloads=[completion for completion, _evaluation in pairs],
                task_id=task_id,
            )
            await store.ingest_eval_payloads(
                payloads=[evaluation for _completion, evaluation in pairs],
                task_id=task_id,
            )
    finally:
        await close_db()


def _persist_judge_checkpoint_batch(
    task_id: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    if not pairs:
        return
    with _LOCK:
        asyncio.run(_persist_judge_checkpoint_batch_async(task_id, pairs))


def _apply_judge(
    rows: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> dict[str, dict[str, float | int]] | None:
    settings = _judge_settings()
    if settings is None:
        return None
    try:
        workers = max(1, int(os.environ.get("HELICOPTER_JUDGE_CONCURRENT_REQUESTS", "10")))
        checkpoint_size = max(1, int(os.environ.get("HELICOPTER_JUDGE_CHECKPOINT_BATCH_SIZE", "50")))
    except ValueError as error:
        raise RuntimeError("judge concurrency and checkpoint settings must be integers") from error

    counts: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0])
    work: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    cached = 0
    for task, (completions, evals) in rows.items():
        for completion, evaluation in zip(completions, evals):
            existing = _stored_judge_result(completion)
            if existing is not None and float(existing["score"]) > 0 and evaluation.get("answer") == "":
                existing = None
            if existing is not None:
                score = _record_judge_result(completion, evaluation, existing)
                counts[task][0] += score
                counts[task][1] += 1
                counts[task][2] += int(score >= 1.0)
                cached += 1
                continue
            stages = completion.get("agent_result", {}).get("model_response", {})
            answer = evaluation.get("answer")
            if answer is None:
                answer = _completion_answer(stages)
            work.append(
                (
                    task,
                    completion,
                    evaluation,
                    {
                        "reference_answer": evaluation.get("ref_answer") or "",
                        "candidate_answer": answer,
                    },
                )
            )

    task_id = os.environ.get("HELICOPTER_SCOREBOARD_TASK_ID", "").strip()
    total = cached + len(work)
    persisted_new = 0
    checkpoint: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def flush_checkpoint() -> None:
        nonlocal persisted_new
        if not task_id or not checkpoint:
            return
        _persist_judge_checkpoint_batch(task_id, list(checkpoint))
        persisted_new += len(checkpoint)
        checkpoint.clear()
        print(f"judge: checkpointed {cached + persisted_new}/{total} rollout(s)")

    if cached:
        print(f"judge: resumed {cached}/{total} rollout(s) from database checkpoints")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_judge_one, item[3], settings): item
            for item in work
        }
        try:
            for future in as_completed(futures):
                task, completion, evaluation, _item = futures[future]
                result = future.result()
                score = _record_judge_result(completion, evaluation, result)
                counts[task][0] += score
                counts[task][1] += 1
                counts[task][2] += int(score >= 1.0)
                checkpoint.append((completion, evaluation))
                if len(checkpoint) >= checkpoint_size:
                    flush_checkpoint()
        finally:
            flush_checkpoint()

    sampling = _sampling_config()
    configured_k = sampling.get("avg_k")
    if configured_k is None:
        raise RuntimeError("judge avg@k requires [lighteval.g1h].avg_k from TOML")
    metric_name = f"judge_avg@{int(configured_k)}"
    return {
        task: {
            metric_name: score_sum / total if total else 0.0,
            "judge_score_sum": score_sum,
            "judge_fully_correct": fully_correct,
            "judge_total": total,
        }
        for task, (score_sum, total, fully_correct) in counts.items()
    }


def _dataset(task: str) -> str:
    parts = [p.strip() for p in str(task).split("|") if p.strip()]
    if len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    name = parts[-1] if parts else str(task)
    return name[len("g1h__") :] if name.startswith("g1h__") else name


def _judge_selected(
    tasks: Iterable[str]
    | Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> bool:
    if isinstance(tasks, Mapping):
        for completions, _evals in tasks.values():
            for completion in completions:
                stats = completion.get("stats")
                metrics = stats.get("metrics") if isinstance(stats, Mapping) else None
                if isinstance(metrics, Mapping) and any(
                    str(key).startswith("deferred_judge_") for key in metrics
                ):
                    return True
        task_names: Iterable[str] = tasks
    else:
        task_names = tasks
    raw = os.environ.get("HELICOPTER_JUDGE_DATASETS")
    if raw is None:
        return True
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    return "*" in selected or any(_dataset(task) in selected for task in task_names)


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
            if pinned_task_id and len(rows) == 1:
                task_id = pinned_task_id
            else:
                task_id = await store.get_or_create_task(
                    job_name=job,
                    job_id=None,
                    dataset=dataset,
                    model=model,
                    is_param_search=False,
                    sampling_config=sampling,
                    config_path=config_path,
                    allow_resume=True,
                )
            if not mark_completed:
                await store.update_task_status(task_id=task_id, status="Running")
            restored = 0
            if _judge_settings() is not None and _judge_selected({task: (completions, evals)}):
                restored = await _restore_judge_checkpoints(task_id, completions, evals)
            if restored:
                print(f"judge: restored {restored} database checkpoint(s) before scoring")
            sample_count = len({int(item["sample_index"]) for item in completions})
            await store.ensure_benchmark_num_samples(dataset=dataset, num_samples=sample_count)
            await store.insert_completion_payloads_batch(payloads=completions, task_id=task_id)
            await store.ingest_eval_payloads(payloads=evals, task_id=task_id)
            await store.record_score_payload(
                task_id=task_id,
                payload={"cot_mode": cot_mode, "metrics": _jsonable(metrics.get(task, {}))},
                mark_completed=mark_completed,
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
                answer = _answer(rollout)
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
                evals.append(
                    {
                        **key,
                        "answer": answer,
                        "ref_answer": _reference(doc),
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
    judge_enabled = _judge_settings() is not None and _judge_selected(rows)
    with _LOCK:
        recorded = asyncio.run(
            _write(
                model,
                "lighteval",
                rows,
                official_metrics,
                mark_completed=not judge_enabled,
            )
        )
    if not judge_enabled:
        return recorded
    judged_metrics = _apply_judge(rows) or {}
    metrics = {task: dict(values) for task, values in official_metrics.items()}
    for task, judge_values in judged_metrics.items():
        metrics.setdefault(str(task), {}).update(_jsonable(judge_values))
    with _LOCK:
        return asyncio.run(_write(model, "lighteval", rows, metrics))


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
        task_id = await store.get_or_create_task(
            job_name="function_calling",
            job_id=None,
            dataset=dataset,
            model=model,
            is_param_search=False,
            sampling_config=_jsonable(sampling_config),
            config_path=config_path,
            allow_resume=True,
        )
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
    task_id: str,
    dataset: str,
    num_samples: int,
    payload: dict[str, Any],
) -> int:
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        await store.ensure_benchmark_num_samples(dataset=dataset, num_samples=num_samples)
        return await store.insert_completion_payloads_batch(
            task_id=task_id,
            payloads=[payload],
        )
    finally:
        await close_db()


def checkpoint_function_calling_result(
    *,
    task_id: str,
    dataset: str,
    num_samples: int,
    sample_index: int,
    sample: Any,
    result: Any,
    sampling_config: Mapping[str, Any],
    root: Path,
    env: Mapping[str, str],
) -> int:
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
        return asyncio.run(
            _checkpoint_function_calling_result(
                task_id=task_id,
                dataset=dataset,
                num_samples=num_samples,
                payload=payload,
            )
        )


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
        task_id = await store.get_or_create_task(
            job_name="lighteval",
            job_id=None,
            dataset=dataset,
            model=model,
            is_param_search=False,
            sampling_config=_sampling_config(),
            config_path=os.environ.get("HELICOPTER_SCOREBOARD_CONFIG_PATH"),
            allow_resume=True,
        )
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
    """Create or resume the stable database task used by both pipeline stages."""

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


def _pending_generation_payloads(
    *,
    task_name: str,
    sample_index: int,
    doc: Any,
    prompt: str,
    repeat_indices: Iterable[int],
    generation_size: int | None = None,
    requested_generation_size: int | None = None,
    prompt_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,
    truncated_prompt_tokens: int = 0,
    context_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build pre-request rows so failed samples retain an exact DB identity."""

    doc_payload = _jsonable(doc)
    if not isinstance(doc_payload, dict):
        doc_payload = {"value": doc_payload}
    sampling_config = _sampling_config()
    if generation_size is not None:
        sampling_config["effective_generation_size"] = int(generation_size)
    if requested_generation_size is not None:
        sampling_config["requested_generation_size"] = int(requested_generation_size)
    if prompt_tokens is not None:
        sampling_config["prompt_tokens"] = int(prompt_tokens)
    if truncate_prompt_tokens is not None:
        sampling_config["truncate_prompt_tokens"] = int(truncate_prompt_tokens)
        sampling_config["truncation_side"] = "left"
    sampling_config["truncated_prompt_tokens"] = int(truncated_prompt_tokens)
    if context_limit is not None:
        sampling_config["context_limit"] = int(context_limit)
    doc_id = doc_payload.get("id")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return [
        {
            "_stage": "generation",
            "status": "Running",
            "sample_index": int(sample_index),
            "repeat_index": int(repeat_index),
            "pass_index": 0,
            "prompt1": prompt,
            "completion1": "",
            "stop_reason1": None,
            "sampling_config": sampling_config,
            "stats": {
                "lighteval_task": task_name,
                "generation_manifest": True,
                "stable_sample_index": int(sample_index),
                "lighteval_doc_id": doc_id,
                "dataset_row_id": doc_id,
                "prompt_sha256": prompt_sha256,
                "identity_contract": "lighteval_sample_index+dataset_row_id+prompt_sha256",
            },
            "agent_result": {"doc": _compact_lighteval_doc(doc_payload)},
            "task_id": doc_id,
        }
        for repeat_index in repeat_indices
    ]


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

    def __init__(self, *, task_id: str, dataset: str, num_samples: int) -> None:
        self.task_id = task_id
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._store: Any | None = None

    async def _open(self) -> None:
        from scoreboard_server.db.connection import init_db
        from scoreboard_server.db.repository import ScoreboardStore
        from scoreboard_server.db.settings import DatabaseSettings

        settings = DatabaseSettings.from_env()
        await init_db(settings, generate_schemas=False)
        self._store = ScoreboardStore(settings=settings)
        await self._store.ensure_benchmark_num_samples(
            dataset=self.dataset,
            num_samples=self.num_samples,
        )

    async def _write(self, payloads: list[dict[str, Any]]) -> int:
        if self._store is None:
            raise RuntimeError("LightEval checkpoint session is not open")
        return await self._store.insert_completion_payloads_batch(
            payloads=payloads,
            task_id=self.task_id,
        )

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
        return self._loop.run_until_complete(self._write(payloads))

    def register_pending(self, requests: Iterable[Mapping[str, Any]]) -> int:
        """Persist every request identity before any endpoint call is submitted."""

        if self._loop is None:
            raise RuntimeError("LightEval checkpoint session is not open")
        payloads: list[dict[str, Any]] = []
        for request in requests:
            payloads.extend(
                _pending_generation_payloads(
                    task_name=str(request["task_name"]),
                    sample_index=int(request["sample_index"]),
                    doc=request["doc"],
                    prompt=str(request["prompt"]),
                    repeat_indices=request["repeat_indices"],
                    generation_size=request.get("generation_size"),
                    requested_generation_size=request.get("requested_generation_size"),
                    prompt_tokens=request.get("prompt_tokens"),
                    truncate_prompt_tokens=request.get("truncate_prompt_tokens"),
                    truncated_prompt_tokens=int(request.get("truncated_prompt_tokens") or 0),
                    context_limit=request.get("context_limit"),
                )
            )
        return self._loop.run_until_complete(self._write(payloads))

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
