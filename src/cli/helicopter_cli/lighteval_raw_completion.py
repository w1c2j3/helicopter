from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import requests
from tqdm import tqdm

from helicopter_cli.lighteval_vllm_sampling import load_sampling_overrides
from lighteval.data import GenerativeTaskDataset
from lighteval.models.endpoints.litellm_model import LiteLLMClient
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import SamplingMethod
from lighteval.utils.cache_management import cached


_TEMPLATE_ENV = "HELICOPTER_PROMPT_TEMPLATE"
_OFFICIAL_USER_STOP = "\nUser:"
_TASK_REQUEST_POLICY_ENV = "HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"


def _strip_prefill_continuation(text: str, template: str | None = None) -> str:
    """Remove the token that only completes an intentionally open think tag.

    The RWKV prompt contracts end in ``<think`` or ``</think`` (without the
    terminal ``>``).  The first generated ``>`` belongs to that prefill, not
    to the assistant answer.  Keeping it changes exact-match results and makes
    otherwise valid unfenced Python syntactically invalid.
    """

    prompt_template = os.environ.get(_TEMPLATE_ENV, "") if template is None else template
    if prompt_template.rstrip().endswith(("<think", "</think")) and text.startswith(">"):
        return text[1:].lstrip()
    return text


def _truncate_official_stops(text: str) -> str:
    """Apply RWKV's text-level stop guards after the endpoint response."""

    positions = [
        position
        for position in (text.find(_OFFICIAL_USER_STOP), text.find("\x00"))
        if position >= 0
    ]
    return text[: min(positions)] if positions else text


def _restore_structured_prefill(
    text: str,
    template: str | None = None,
    *,
    restore_stop_fence: bool = False,
) -> str:
    """Restore fenced prefixes and an API-omitted configured closing fence."""

    prompt_template = os.environ.get(_TEMPLATE_ENV, "") if template is None else template
    stripped = prompt_template.rstrip()
    for language in ("python", "diff", ""):
        fence = f"```{language}"
        if stripped.endswith(fence):
            restored = f"{fence}\n{text.lstrip()}"
            if restore_stop_fence and "```" not in text:
                restored = restored.rstrip() + "\n```"
            return restored
    return text


def _canonical_task_name(task_name: str | None) -> str:
    name = str(task_name or "").split("|", 1)[0]
    return name[len("g1h__") :] if name.startswith("g1h__") else name


def _task_request_policy(task_name: str | None) -> dict[str, Any]:
    raw = os.environ.get(_TASK_REQUEST_POLICY_ENV, "").strip()
    if not raw:
        return {}
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {_TASK_REQUEST_POLICY_ENV}: {error}") from error
    if not isinstance(policy, dict):
        raise RuntimeError(f"{_TASK_REQUEST_POLICY_ENV} must contain a JSON object")
    tasks = policy.get("tasks")
    if not isinstance(tasks, dict):
        return {}
    canonical = _canonical_task_name(task_name)
    entry = tasks.get(canonical)
    if entry is None:
        parent_matches = [
            (configured_task, candidate)
            for configured_task, candidate in tasks.items()
            if canonical.startswith(f"{configured_task}:")
        ]
        entry = (
            max(parent_matches, key=lambda item: len(item[0]))[1]
            if parent_matches
            else {}
        )
    return entry if isinstance(entry, dict) else {}


def _merge_stops(task_stops: list[str] | None, configured: Any, *, inherit: bool) -> list[str] | None:
    values: list[str] = []
    if inherit:
        values.extend(str(item) for item in (task_stops or []) if str(item))
    if configured is not None:
        if not isinstance(configured, list):
            raise RuntimeError("task stop policy must be a JSON array")
        values.extend(str(item) for item in configured if str(item))
    return list(dict.fromkeys(values)) or None


def _configured_stops(task_name: str | None, task_stops: list[str] | None) -> list[str] | None:
    policy = _task_request_policy(task_name)
    if not policy:
        return task_stops
    prompt_mode = str(policy.get("prompt_mode") or "").strip().lower()
    if prompt_mode.startswith("naive"):
        return [_OFFICIAL_USER_STOP]
    return _merge_stops(
        task_stops,
        policy.get("stop"),
        inherit=bool(policy.get("inherit_task_stops", True)),
    )


def _configured_sampling(task_name: str | None) -> dict[str, Any]:
    value = _task_request_policy(task_name).get("sampling", {})
    if not isinstance(value, dict):
        raise RuntimeError("task sampling policy must be a JSON object")
    return dict(value)


def _effective_context_budget(task_name: str | None, default: int | None) -> int | None:
    value = _configured_sampling(task_name).get("context_budget", default)
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"context_budget must be a positive integer, got {value!r}") from error
    if result <= 0:
        raise RuntimeError(f"context_budget must be a positive integer, got {result}")
    return result


def _configured_prompt_template(task_name: str | None, default: str) -> str:
    value = _task_request_policy(task_name).get("prompt_template", default)
    if not isinstance(value, str) or "{query}" not in value:
        raise RuntimeError("task prompt_template must be a string containing {query}")
    return value


def _render_prompt(template: str, query: str, *, task_name: str | None = None) -> str:
    """Render the benchmark-owned template without assembling hidden context."""
    return template.format(query=str(query))


def _validate_raw_question_contract(task_name: str | None, prompt: str) -> None:
    policy = _task_request_policy(task_name)
    if not policy.get("benchmark_config_path"):
        return
    lowered = prompt.casefold()
    forbidden_cues = (
        "answer the following multiple choice question",
        "the following are multiple choice questions",
        "think step by step before answering",
        "let's think step by step",
        "solve the problem using one clean solution path",
    )
    found = next((cue for cue in forbidden_cues if cue in lowered), None)
    if found is not None:
        raise RuntimeError(
            f"benchmark {_canonical_task_name(task_name)!r} violates the raw-question "
            f"contract with an added cue: {found!r}"
        )
    if re.search(r"\nanswer:\s*\n+assistant:", prompt, flags=re.IGNORECASE):
        raise RuntimeError(
            f"benchmark {_canonical_task_name(task_name)!r} violates the raw-question "
            "contract with an added trailing Answer: cue"
        )


def _configured_multi_turn_template(task_name: str | None) -> str | None:
    value = _task_request_policy(task_name).get("multi_turn_template")
    if value is None:
        return None
    required = ("{first_prompt}", "{first_answer}", "{query}")
    if not isinstance(value, str) or any(field not in value for field in required):
        raise RuntimeError(
            "task multi_turn_template must contain {first_prompt}, {first_answer}, and {query}"
        )
    return value


def _multi_turn_queries(doc: Any) -> list[str]:
    specific = getattr(doc, "specific", None)
    value = specific.get("multi_turn_queries") if isinstance(specific, dict) else None
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError("LightEval multi-turn generation currently requires exactly two turns")
    queries = [str(item).strip() for item in value]
    if any(not item for item in queries):
        raise RuntimeError("LightEval multi-turn queries must be non-empty strings")
    return queries


def _completion_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("raw completion mode requires a base_url")
    base = base_url.rstrip("/")
    return f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"


def _tokenize_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("raw completion mode requires a base_url")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/tokenize"


def _served_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _api_headers(self: LiteLLMClient) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("HELICOPTER_EVAL_API_KEY") or self.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _effective_max_tokens(task_name: str | None, default: int | None) -> int | None:
    global_sampling = load_sampling_overrides()
    value = global_sampling.get("max_tokens", default)
    task_sampling = _configured_sampling(task_name)
    return task_sampling.get("max_tokens", value)


@dataclass(frozen=True)
class _RequestContextFit:
    max_tokens: int | None
    truncate_prompt_tokens: int | None
    prompt_tokens: int | None
    context_limit: int

    @property
    def truncated_prompt_tokens(self) -> int:
        if self.prompt_tokens is None or self.truncate_prompt_tokens is None:
            return 0
        return max(self.prompt_tokens - self.truncate_prompt_tokens, 0)


def _fit_request_to_context(
    self: LiteLLMClient,
    *,
    prompt: str,
    requested_max_tokens: int | None,
    context_budget: int | None = None,
) -> _RequestContextFit:
    """Preserve the full official prompt and fit generation into remaining context."""

    max_model_length = int(self.max_length)
    if context_budget is not None:
        context_budget = int(context_budget)
        if context_budget <= 0:
            raise RuntimeError(f"context_budget must be positive, got {context_budget}")
        max_model_length = min(max_model_length, context_budget)
    if requested_max_tokens is None:
        return _RequestContextFit(None, None, None, max_model_length)
    requested = int(requested_max_tokens)
    if requested <= 0:
        raise RuntimeError(f"max_tokens must be positive, got {requested}")
    if requested >= max_model_length:
        raise RuntimeError(
            f"max_tokens is {requested} but model context is {max_model_length}; "
            "at least one prompt token must remain"
        )
    # RWKV token count is bounded by UTF-8 bytes plus BOS. Only potentially
    # tight prompts need an authoritative endpoint tokenization round trip.
    if len(prompt.encode("utf-8")) + 1 + requested <= max_model_length:
        return _RequestContextFit(requested, None, None, max_model_length)

    response = requests.post(
        _tokenize_url(self.base_url),
        headers=_api_headers(self),
        json={"model": _served_model(self.model), "prompt": prompt},
        timeout=self.timeout or 180,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(
            f"tokenize endpoint returned HTTP {response.status_code}: {response.text[:2_000]}"
        ) from error
    body = response.json()
    prompt_tokens = int(body["count"])
    endpoint_max = int(body.get("max_model_len") or max_model_length)
    context_limit = min(max_model_length, endpoint_max)
    remaining_generation_tokens = context_limit - prompt_tokens
    if remaining_generation_tokens < 1:
        raise RuntimeError(
            f"prompt uses {prompt_tokens} tokens but endpoint context is {context_limit}; "
            "the official prompt cannot be preserved"
        )
    return _RequestContextFit(
        max_tokens=min(requested, remaining_generation_tokens),
        truncate_prompt_tokens=None,
        prompt_tokens=prompt_tokens,
        context_limit=context_limit,
    )


def _request(
    self: LiteLLMClient,
    prompt: str,
    max_tokens: int | None,
    num_samples: int,
    stops: list[str] | None,
    *,
    prompt_template: str | None = None,
    force_max_tokens: bool = False,
    force_stops: bool = False,
    truncate_prompt_tokens: int | None = None,
    task_name: str | None = None,
) -> ModelResponse:
    overrides = load_sampling_overrides()
    configured_stop = overrides.pop("stop", None)
    overrides.pop("max_tokens", None)
    overrides.update(_configured_sampling(task_name))
    overrides.pop("max_tokens", None)
    overrides.pop("context_budget", None)
    task_policy = _task_request_policy(task_name)
    task_policy_stop = _configured_stops(task_name, stops)
    effective_stop = task_policy_stop if task_policy else (
        configured_stop if configured_stop is not None else stops
    )
    payload: dict[str, Any] = {
        "model": _served_model(self.model),
        "prompt": prompt,
        "max_tokens": max_tokens if force_max_tokens else _effective_max_tokens(task_name, max_tokens),
        "n": int(num_samples),
        "stop": stops if force_stops else effective_stop,
        "truncate_prompt_tokens": truncate_prompt_tokens,
        "truncation_side": "left" if truncate_prompt_tokens is not None else None,
        **overrides,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    headers = _api_headers(self)
    last_error: Exception | None = None
    for attempt in range(self.API_MAX_RETRY):
        try:
            response = requests.post(
                _completion_url(self.base_url),
                headers=headers,
                json=payload,
                timeout=self.timeout or 180,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                response_body = response.text.strip()
                if len(response_body) > 2_000:
                    response_body = f"{response_body[:2_000]}..."
                raise RuntimeError(
                    f"completion endpoint returned HTTP {response.status_code}: "
                    f"{response_body or '<empty response body>'}"
                ) from error
            body = response.json()
            choices = sorted(body.get("choices") or [], key=lambda item: item.get("index", 0))
            if not choices:
                raise RuntimeError(f"completion response has no choices: {body!r}")
            raw_texts = [
                _truncate_official_stops(str(choice.get("text") or ""))
                for choice in choices
            ]
            finish_reasons = [choice.get("finish_reason") for choice in choices]
            configured_fence_stop = "\n```" in (payload.get("stop") or [])
            texts = [
                _restore_structured_prefill(
                    _strip_prefill_continuation(text, prompt_template),
                    prompt_template,
                    restore_stop_fence=(configured_fence_stop and finish_reasons[index] == "stop"),
                )
                for index, text in enumerate(raw_texts)
            ]
            result = ModelResponse(text=texts, input=prompt)
            result.raw_text = raw_texts
            result.finish_reason = finish_reasons[0] if len(finish_reasons) == 1 else finish_reasons
            result.usage = body.get("usage")
            return result
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)))
    raise RuntimeError(f"raw completion request failed after retries: {last_error}")



def _one(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _response_from_rollouts(rollouts: list[dict[str, Any]]) -> ModelResponse:
    if not rollouts:
        raise RuntimeError("cannot rebuild an empty response")
    def restored_text(item: dict[str, Any], key: str) -> str:
        return str(_one(item.get(key)) or "")

    post_processed = [
        restored_text(item, "text_post_processed")
        if _one(item.get("text_post_processed")) is not None
        else None
        for item in rollouts
    ]
    response = ModelResponse(
        input=rollouts[0].get("input"),
        text=[restored_text(item, "text") for item in rollouts],
        text_post_processed=(
            [str(value or "") for value in post_processed]
            if any(value is not None for value in post_processed)
            else None
        ),
    )
    response.raw_text = [
        restored_text(item, "raw_text")
        if _one(item.get("raw_text")) is not None
        else restored_text(item, "text")
        for item in rollouts
    ]
    response.finish_reason = [_one(item.get("finish_reason")) for item in rollouts]
    response.usage = [item.get("usage") for item in rollouts]
    stages = [item.get("stages") for item in rollouts]
    if any(stage is not None for stage in stages):
        response.stages_by_rollout = stages
    return response


def _stored_generation(task_id: str | None) -> dict[int, dict[int, dict[str, Any]]]:
    if not task_id:
        return {}
    from helicopter_cli.scoreboard_bridge import load_lighteval_generation

    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for row in load_lighteval_generation(task_id=task_id):
        if str(row.get("status") or "").lower() != "completed":
            continue
        context = row.get("context")
        if not isinstance(context, dict):
            continue
        agent_result = context.get("agent_result")
        if not isinstance(agent_result, dict):
            continue
        model_response = agent_result.get("model_response")
        if not isinstance(model_response, dict):
            continue
        stats = context.get("stats") if isinstance(context.get("stats"), dict) else {}
        doc = agent_result.get("doc") if isinstance(agent_result.get("doc"), dict) else {}
        stages = context.get("stages") if isinstance(context.get("stages"), list) else []
        first_stage = stages[0] if stages and isinstance(stages[0], dict) else {}
        grouped.setdefault(int(row["sample_index"]), {})[int(row["repeat_index"])] = {
            "model_response": model_response,
            "dataset_row_id": stats.get(
                "dataset_row_id",
                stats.get("lighteval_doc_id", doc.get("id", context.get("task_id"))),
            ),
            "prompt": first_stage.get("prompt"),
        }
    return grouped


def _doc_id(doc: Any) -> Any:
    if isinstance(doc, dict):
        return doc.get("id")
    return getattr(doc, "id", None)


def _identity_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _validated_stored_rollouts(
    checkpoints: dict[int, dict[str, Any]],
    *,
    sample_index: int,
    doc: Any,
    prompt: str,
) -> dict[int, dict[str, Any]]:
    """Reject stale checkpoints whose raw dataset identity no longer matches."""

    current_doc_id = _doc_id(doc)
    rollouts: dict[int, dict[str, Any]] = {}
    for repeat_index, checkpoint in checkpoints.items():
        stored_doc_id = checkpoint.get("dataset_row_id")
        stored_prompt = checkpoint.get("prompt")
        if stored_doc_id is None and not isinstance(stored_prompt, str):
            raise RuntimeError(
                f"generation checkpoint {sample_index}/{repeat_index} has no dataset identity; "
                "refusing unsafe index-only resume"
            )
        if (
            stored_doc_id is not None
            and current_doc_id is not None
            and _identity_value(stored_doc_id) != _identity_value(current_doc_id)
        ):
            raise RuntimeError(
                f"generation checkpoint {sample_index}/{repeat_index} belongs to dataset row "
                f"{stored_doc_id!r}, not current row {current_doc_id!r}; refusing mismatched resume"
            )
        if isinstance(stored_prompt, str) and stored_prompt != prompt:
            raise RuntimeError(
                f"generation checkpoint {sample_index}/{repeat_index} prompt changed for dataset "
                f"row {current_doc_id!r}; refusing mismatched resume"
            )
        model_response = checkpoint.get("model_response")
        if not isinstance(model_response, dict):
            raise RuntimeError(
                f"generation checkpoint {sample_index}/{repeat_index} has no model response"
            )
        rollouts[int(repeat_index)] = model_response
    return rollouts


def _response_rollouts(response: ModelResponse) -> list[dict[str, Any]]:
    from helicopter_cli.scoreboard_bridge import _response_payload, _rollout_count, _rollout_response

    payload = _response_payload(response)
    return [_rollout_response(payload, index) for index in range(_rollout_count(payload))]


def _checkpoint_response(
    *,
    task_id: str | None,
    dataset_name: str,
    task_name: str,
    total_samples: int,
    sample_index: int,
    doc: Any,
    response: ModelResponse,
    repeat_indices: list[int],
    generation_size: int,
    checkpoint_session: Any | None = None,
) -> None:
    if checkpoint_session is not None:
        checkpoint_session.checkpoint(
            task_name=task_name,
            sample_index=sample_index,
            doc=doc,
            response=response,
            repeat_indices=repeat_indices,
            generation_size=generation_size,
        )
        return
    if not task_id:
        return
    from helicopter_cli.scoreboard_bridge import checkpoint_lighteval_response

    checkpoint_lighteval_response(
        task_id=task_id,
        dataset=dataset_name,
        task_name=task_name,
        num_samples=total_samples,
        sample_index=sample_index,
        doc=doc,
        response=response,
        repeat_indices=repeat_indices,
        generation_size=generation_size,
    )


def _merge_response_rollouts(
    *,
    existing: dict[int, dict[str, Any]],
    generated: ModelResponse | None,
    generated_indices: list[int],
    rollout_n: int,
) -> ModelResponse:
    merged = dict(existing)
    if generated is not None:
        generated_rollouts = _response_rollouts(generated)
        if len(generated_rollouts) != len(generated_indices):
            raise RuntimeError(
                f"completion count mismatch: got {len(generated_rollouts)}, expected {len(generated_indices)}"
            )
        merged.update(zip(generated_indices, generated_rollouts))
    missing = [index for index in range(rollout_n) if index not in merged]
    if missing:
        raise RuntimeError(f"generation checkpoint is incomplete after request: missing repeats {missing}")
    return _response_from_rollouts([merged[index] for index in range(rollout_n)])

def _generate_two_turn_response(
    self: LiteLLMClient,
    *,
    doc: Any,
    task_name: str,
    first_prompt: str,
    queries: list[str],
    max_tokens: int | None,
    missing_count: int,
    stops: list[str] | None,
    first_template: str,
    first_context_fit: _RequestContextFit,
    context_budget: int | None = None,
) -> ModelResponse:
    multi_turn_template = _configured_multi_turn_template(task_name)
    if multi_turn_template is None:
        raise RuntimeError(
            f"task {task_name!r} has multi_turn_queries but no TOML multi_turn_template"
        )
    first = _request(
        self,
        first_prompt,
        first_context_fit.max_tokens,
        missing_count,
        stops,
        prompt_template=first_template,
        force_max_tokens=True,
        truncate_prompt_tokens=first_context_fit.truncate_prompt_tokens,
        task_name=task_name,
    )
    first_rollouts = _response_rollouts(first)
    if len(first_rollouts) != missing_count:
        raise RuntimeError(
            f"first-turn completion count mismatch: got {len(first_rollouts)}, expected {missing_count}"
        )

    texts: list[str] = []
    raw_texts: list[str] = []
    finish_reasons: list[Any] = []
    usages: list[Any] = []
    stages_by_rollout: list[list[dict[str, Any]]] = []
    for first_rollout in first_rollouts:
        first_answer = str(_one(first_rollout.get("text")) or "")
        second_prompt = multi_turn_template.format(
            first_prompt=first_prompt,
            first_answer=first_answer,
            query=queries[1],
        )
        second_context_fit = _fit_request_to_context(
            self,
            prompt=second_prompt,
            requested_max_tokens=max_tokens,
            context_budget=context_budget,
        )
        second = _request(
            self,
            second_prompt,
            second_context_fit.max_tokens,
            1,
            stops,
            prompt_template=multi_turn_template,
            force_max_tokens=True,
            truncate_prompt_tokens=second_context_fit.truncate_prompt_tokens,
            task_name=task_name,
        )
        second_rollout = _response_rollouts(second)[0]
        second_answer = str(_one(second_rollout.get("text")) or "")
        texts.append(second_answer)
        raw_texts.append(str(_one(second_rollout.get("raw_text")) or second_answer))
        finish_reasons.append(_one(second_rollout.get("finish_reason")))
        usages.append({"stages": [first_rollout.get("usage"), second_rollout.get("usage")]})
        stages_by_rollout.append(
            [
                {
                    "prompt": first_prompt,
                    "completion": first_answer,
                    "stop_reason": _one(first_rollout.get("finish_reason")),
                },
                {
                    "prompt": second_prompt,
                    "completion": second_answer,
                    "stop_reason": _one(second_rollout.get("finish_reason")),
                },
            ]
        )
    response = ModelResponse(text=texts, input=first_prompt)
    response.raw_text = raw_texts
    response.finish_reason = finish_reasons
    response.usage = usages
    response.stages_by_rollout = stages_by_rollout
    return response


@cached(SamplingMethod.GENERATIVE)
def greedy_until(self: LiteLLMClient, docs: list[Any]) -> list[ModelResponse]:
    """Generate with PostgreSQL checkpoints as the only response cache.

    Helicopter configures LightEval with use_cache=false, so the native cache
    decorator delegates directly to this method. Resume is handled exclusively
    by _stored_generation and every fresh response reaches the DB checkpoint.
    """

    dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
    default_template = os.environ[_TEMPLATE_ENV]
    task_id = os.environ.get("HELICOPTER_SCOREBOARD_TASK_ID", "").strip() or None
    dataset_name = os.environ.get("HELICOPTER_SCOREBOARD_DATASET", "").strip()
    stored = _stored_generation(task_id)
    original_indices = {id(doc): index for index, doc in enumerate(docs)}
    results: list[ModelResponse] = []

    if task_id or dataset_name:
        from helicopter_cli.scoreboard_bridge import LightevalCheckpointSession

        checkpoint_context = LightevalCheckpointSession(
            task_id=task_id,
            dataset=dataset_name,
            num_samples=len(docs),
        )
    else:
        checkpoint_context = nullcontext(None)

    with checkpoint_context as checkpoint_session:
        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            split_docs = list(split)
            contexts = [str(getattr(doc, "query", "")) for doc in split_docs]
            task_names = [str(getattr(doc, "task_name", "") or dataset_name) for doc in split_docs]
            templates = [
                _configured_prompt_template(task_name, default_template)
                for task_name in task_names
            ]
            prompts = [
                _render_prompt(template, context, task_name=task_name)
                for template, context, task_name in zip(templates, contexts, task_names)
            ]
            for task_name, prompt_text in zip(task_names, prompts):
                _validate_raw_question_contract(task_name, prompt_text)
            max_tokens = split[0].generation_size
            rollout_n = int(split[0].num_samples)
            stops = split[0].stop_sequences
            responses: list[ModelResponse | None] = [None] * len(split_docs)

            work_items: list[
                tuple[int, int, list[int], dict[int, dict[str, Any]], int | None, _RequestContextFit]
            ] = []
            for position, doc in enumerate(split_docs):
                sample_index = original_indices.get(id(doc))
                if sample_index is None:
                    raise RuntimeError("LightEval changed document identity; cannot assign a stable sample_index")
                existing = _validated_stored_rollouts(
                    stored.get(sample_index, {}),
                    sample_index=sample_index,
                    doc=doc,
                    prompt=prompts[position],
                )
                missing = [index for index in range(rollout_n) if index not in existing]
                if not missing:
                    responses[position] = _merge_response_rollouts(
                        existing=existing,
                        generated=None,
                        generated_indices=[],
                        rollout_n=rollout_n,
                    )
                    continue
                requested_max_tokens = _effective_max_tokens(task_names[position], max_tokens)
                context_budget = _effective_context_budget(task_names[position], int(self.max_length))
                context_fit = _fit_request_to_context(
                    self,
                    prompt=prompts[position],
                    requested_max_tokens=requested_max_tokens,
                    context_budget=context_budget,
                )
                work_items.append(
                    (
                        position,
                        sample_index,
                        missing,
                        existing,
                        requested_max_tokens,
                        context_fit,
                    )
                )

            def generate_one(
                position: int,
                missing: list[int],
                context_fit: _RequestContextFit,
                context_budget: int | None,
            ) -> ModelResponse:
                queries = _multi_turn_queries(split_docs[position])
                if queries:
                    return _generate_two_turn_response(
                        self,
                        doc=split_docs[position],
                        task_name=task_names[position],
                        first_prompt=prompts[position],
                        queries=queries,
                        max_tokens=context_fit.max_tokens,
                        missing_count=len(missing),
                        stops=stops,
                        first_template=templates[position],
                        first_context_fit=context_fit,
                        context_budget=context_budget,
                    )
                response = _request(
                    self,
                    prompts[position],
                    context_fit.max_tokens,
                    len(missing),
                    stops,
                    prompt_template=templates[position],
                    force_max_tokens=True,
                    truncate_prompt_tokens=context_fit.truncate_prompt_tokens,
                    task_name=task_names[position],
                )
                return response

            with ThreadPoolExecutor(self.concurrent_requests) as executor:
                futures: dict[
                    Any,
                    tuple[int, int, list[int], dict[int, dict[str, Any]], _RequestContextFit],
                ] = {}
                for (
                    position,
                    sample_index,
                    missing,
                    existing,
                    _requested_max_tokens,
                    context_fit,
                ) in work_items:
                    future = executor.submit(
                        generate_one,
                        position,
                        missing,
                        context_fit,
                        _effective_context_budget(task_names[position], int(self.max_length)),
                    )
                    futures[future] = (position, sample_index, missing, existing, context_fit)

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Raw completions",
                    position=1,
                    leave=False,
                    disable=self.disable_tqdm,
                ):
                    position, sample_index, missing, existing, context_fit = futures[future]
                    generated = future.result()
                    doc = split_docs[position]
                    task_name = str(getattr(doc, "task_name", "") or dataset_name)
                    _checkpoint_response(
                        task_id=task_id,
                        dataset_name=dataset_name or task_name,
                        task_name=task_name,
                        total_samples=len(docs),
                        sample_index=sample_index,
                        doc=doc,
                        response=generated,
                        repeat_indices=missing,
                        generation_size=int(context_fit.max_tokens or max_tokens),
                        checkpoint_session=checkpoint_session,
                    )
                    responses[position] = _merge_response_rollouts(
                        existing=existing,
                        generated=generated,
                        generated_indices=missing,
                        rollout_n=rollout_n,
                    )

            if any(response is None for response in responses):
                raise RuntimeError("generation returned an incomplete response set")
            results.extend(response for response in responses if response is not None)

    return dataset.get_original_order(results)


LiteLLMClient.greedy_until = greedy_until
