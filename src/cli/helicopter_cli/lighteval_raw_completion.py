from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from string import ascii_uppercase
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
_STRIP_FLOWER_ENV = "HELICOPTER_STRIP_TERMINAL_FLOWER"
_MATH_FINAL_SUFFIX_ENV = "HELICOPTER_MATH_FINAL_SUFFIX"
_MATH_FINAL_MAX_TOKENS_ENV = "HELICOPTER_MATH_FINAL_MAX_TOKENS"


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


def _choice_mapping(choices: Any) -> dict[str, str]:
    if not isinstance(choices, (list, tuple)):
        return {}
    normalized = [str(choice).strip().upper() for choice in choices]
    expected = list(ascii_uppercase[: len(normalized)])
    if len(normalized) < 2 or normalized != expected:
        return {}
    return {label: str(choice) for label, choice in zip(normalized, choices)}


_CHOICE_PATTERNS = (
    re.compile(r"\\boxed\{\s*(?:\\(?:text|mathrm)\{\s*)?([A-Z])", re.IGNORECASE),
    re.compile(
        r"(?:final\s+)?(?:the\s+)?(?:correct\s+)?answer\s*"
        r"(?:is|:)?\s*(?:option\s*)?[\[(]?\s*([A-Z])\b",
        re.IGNORECASE,
    ),
)


def _extract_choice(text: str, allowed: set[str]) -> str:
    candidates: list[tuple[int, str]] = []
    for pattern in _CHOICE_PATTERNS:
        for match in pattern.finditer(text):
            label = match.group(1).upper()
            if label in allowed:
                candidates.append((match.start(), label))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    for line in reversed(text.splitlines()):
        match = re.fullmatch(r"\s*[\[(]?\s*([A-Z])\s*[\])]?[.\s]*", line, re.IGNORECASE)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()

    matches = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text.upper())
    return next((label for label in reversed(matches) if label in allowed), "")


def _postprocess_choice_response(doc: Any, response: ModelResponse) -> ModelResponse:
    mapping = _choice_mapping(getattr(doc, "choices", None))
    if not mapping:
        return response
    processed: list[str] = []
    for text in response.text:
        label = _extract_choice(str(text or ""), set(mapping))
        processed.append(mapping[label] if label else str(text or ""))
    response.text_post_processed = processed
    return response


def _uses_math_final_stage(doc: Any) -> bool:
    suffix = os.environ.get(_MATH_FINAL_SUFFIX_ENV, "")
    choices = getattr(doc, "choices", None)
    return bool(suffix and isinstance(choices, (list, tuple)) and len(choices) == 1 and str(choices[0]).strip())


def _with_closed_think(text: str) -> str:
    stripped = text.rstrip()
    return stripped if "</think>" in stripped else f"{stripped}\n</think>"


def _usage_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _combined_usage(stage1: Any, stage2: Any) -> dict[str, Any]:
    first, second = _usage_dict(stage1), _usage_dict(stage2)
    combined: dict[str, Any] = {"stage1": first, "stage2": second}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [item.get(key) for item in (first, second)]
        if all(isinstance(value, int) for value in values):
            combined[key] = sum(values)
    return combined


def _math_final_response(self: LiteLLMClient, prompt: str, doc: Any, response: ModelResponse) -> ModelResponse:
    if not _uses_math_final_stage(doc):
        return response
    suffix = os.environ[_MATH_FINAL_SUFFIX_ENV]
    try:
        final_max_tokens = max(1, int(os.environ.get(_MATH_FINAL_MAX_TOKENS_ENV, "64")))
    except ValueError as error:
        raise RuntimeError(f"invalid {_MATH_FINAL_MAX_TOKENS_ENV}") from error

    raw_stage1 = list(getattr(response, "raw_text", response.text))
    stage1_finish = getattr(response, "finish_reason", None)
    stage1_usage = getattr(response, "usage", None)
    combined_texts: list[str] = []
    combined_raw_texts: list[str] = []
    final_reasons: list[Any] = []
    final_usages: list[Any] = []
    recorded_stages: list[dict[str, Any]] = []

    for index, (clean_text, raw_text) in enumerate(zip(response.text, raw_stage1)):
        raw_reasoning = _with_closed_think(str(raw_text or ""))
        clean_reasoning = _with_closed_think(str(clean_text or ""))
        final_prompt = f"{prompt}{raw_reasoning}{suffix}"
        final_response = _request(
            self,
            final_prompt,
            final_max_tokens,
            1,
            ["}"],
            prompt_template=suffix,
            force_max_tokens=True,
            force_stops=True,
        )
        final_piece = str(final_response.text[0] if final_response.text else "").strip()
        combined = f"{clean_reasoning}{suffix}{final_piece}"
        raw_combined = f"{raw_reasoning}{suffix}{final_piece}"
        if suffix.rstrip().endswith("{") and not combined.rstrip().endswith("}"):
            combined += "}"
            raw_combined += "}"
        combined_texts.append(combined)
        combined_raw_texts.append(raw_combined)
        final_reasons.append(getattr(final_response, "finish_reason", None))
        final_usages.append(getattr(final_response, "usage", None))
        if len(response.text) == 1:
            first_reason = stage1_finish[0] if isinstance(stage1_finish, list) and stage1_finish else stage1_finish
            recorded_stages = [
                {"prompt": prompt, "completion": str(clean_text or ""), "stop_reason": first_reason},
                {"prompt": final_prompt, "completion": final_piece, "stop_reason": final_reasons[-1]},
            ]

    response.text = combined_texts
    response.text_post_processed = None
    response.raw_text = combined_raw_texts
    response.finish_reason = final_reasons[0] if len(final_reasons) == 1 else final_reasons
    if len(final_usages) == 1:
        response.usage = _combined_usage(stage1_usage, final_usages[0])
    else:
        response.usage = {"stage1": stage1_usage, "stage2": final_usages}
    if recorded_stages:
        response.stages = recorded_stages
    return response


def _completion_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("raw completion mode requires a base_url")
    base = base_url.rstrip("/")
    return f"{base}/completions" if base.endswith("/v1") else f"{base}/v1/completions"


def _served_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _strip_terminal_flower(text: str) -> str:
    if os.environ.get(_STRIP_FLOWER_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return text
    return re.sub(r"✿\s*$", "", text)


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
) -> ModelResponse:
    overrides = load_sampling_overrides()
    configured_stop = overrides.pop("stop", None)
    configured_max = overrides.pop("max_tokens", None)
    payload: dict[str, Any] = {
        "model": _served_model(self.model),
        "prompt": prompt,
        "max_tokens": max_tokens if force_max_tokens else (configured_max if configured_max is not None else max_tokens),
        "n": int(num_samples),
        "stop": stops if force_stops else (configured_stop if configured_stop is not None else stops),
        **overrides,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("HELICOPTER_EVAL_API_KEY") or self.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_error: Exception | None = None
    for attempt in range(self.API_MAX_RETRY):
        try:
            response = requests.post(
                _completion_url(self.base_url),
                headers=headers,
                json=payload,
                timeout=self.timeout or 180,
            )
            response.raise_for_status()
            body = response.json()
            choices = sorted(body.get("choices") or [], key=lambda item: item.get("index", 0))
            if not choices:
                raise RuntimeError(f"completion response has no choices: {body!r}")
            raw_texts = [_strip_terminal_flower(str(choice.get("text") or "")) for choice in choices]
            texts = [_strip_prefill_continuation(text, prompt_template) for text in raw_texts]
            result = ModelResponse(text=texts, input=prompt)
            result.raw_text = raw_texts
            finish_reasons = [choice.get("finish_reason") for choice in choices]
            result.finish_reason = finish_reasons[0] if len(finish_reasons) == 1 else finish_reasons
            result.usage = body.get("usage")
            return result
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)))
    raise RuntimeError(f"raw completion request failed after retries: {last_error}")


@cached(SamplingMethod.GENERATIVE)
def greedy_until(self: LiteLLMClient, docs: list[Any]) -> list[ModelResponse]:
    dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
    template = os.environ[_TEMPLATE_ENV]
    results: list[ModelResponse] = []
    for split in tqdm(
        dataset.splits_iterator(),
        total=dataset.num_dataset_splits,
        desc="Splits",
        position=0,
        disable=self.disable_tqdm,
    ):
        split_docs = list(split)
        contexts = [self.prompt_manager._prepare_plain_text(doc) for doc in split_docs]
        prompts = [template.format(query=context) for context in contexts]
        max_tokens = split[0].generation_size
        num_samples = split[0].num_samples
        stops = split[0].stop_sequences
        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            responses = list(
                tqdm(
                    executor.map(
                        lambda prompt: _request(
                            self,
                            prompt,
                            max_tokens,
                            num_samples,
                            stops,
                            prompt_template=template,
                        ),
                        prompts,
                    ),
                    total=len(prompts),
                    desc="Raw completions",
                    position=1,
                    leave=False,
                    disable=self.disable_tqdm,
                )
            )
        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            finalized = list(
                executor.map(
                    lambda item: _math_final_response(self, item[0], item[1], item[2]),
                    zip(prompts, split_docs, responses),
                )
            )
        results.extend(_postprocess_choice_response(doc, response) for doc, response in zip(split_docs, finalized))
    return dataset.get_original_order(results)


LiteLLMClient.greedy_until = greedy_until
