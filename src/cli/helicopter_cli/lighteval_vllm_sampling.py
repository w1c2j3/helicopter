from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


VLLM_SAMPLING_ENV = "HELICOPTER_VLLM_SAMPLING_JSON"
_PATCH_MARKER = "_helicopter_vllm_sampling_patched"


def load_sampling_overrides(value: str | None = None) -> dict[str, Any]:
    raw = value if value is not None else os.environ.get(VLLM_SAMPLING_ENV)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {VLLM_SAMPLING_ENV}: {error}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{VLLM_SAMPLING_ENV} must contain a JSON object")
    return dict(payload)


def merge_sampling_kwargs(
    kwargs: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply configured vLLM fields to a LiteLLM completion call.

    LiteLLM's LightEval adapter currently drops ``top_k``, ``min_p``,
    ``presence_penalty`` and ``penalty_decay`` before the request reaches an
    OpenAI-compatible endpoint. The local vLLM endpoint accepts these fields,
    so this compatibility layer restores them without modifying either
    dependency's installed package.
    """
    merged = dict(kwargs)
    for key, value in overrides.items():
        if key == "max_tokens":
            merged["max_tokens"] = value
            merged["max_completion_tokens"] = value
        else:
            merged[key] = value
    return merged


def patch_litellm_sampling() -> None:
    overrides = load_sampling_overrides()
    if not overrides:
        return

    import litellm

    if getattr(litellm.completion, _PATCH_MARKER, False):
        return
    original_completion = litellm.completion

    def completion(*args: Any, **kwargs: Any):
        return original_completion(
            *args,
            **merge_sampling_kwargs(kwargs, overrides),
        )

    setattr(completion, _PATCH_MARKER, True)
    litellm.completion = completion


patch_litellm_sampling()
