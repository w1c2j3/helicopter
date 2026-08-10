"""Strict TOML profiles for raw RWKV completion evaluations.

The profile is deliberately small: it owns task grouping, the RWKV prompt
wrapper, rollout count, and every field sent to vllm-rwkv's completions API.
Task prompt functions remain responsible only for extracting raw dataset text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


_PROFILE_FIELDS = {"name", "model", "tasks", "adapter", "raw_input"}
_PROMPT_FIELDS = {"mode", "template"}
_EVALUATION_FIELDS = {"num_samples", "metric"}
_TOP_LEVEL_FIELDS = {
    "model_catalog",
    "models",
    "runtime",
    "paths",
    "profile",
    "prompt",
    "evaluation",
    "sampling",
    "lighteval",
}
RWKV_PROFILE_ADAPTERS = frozenset({"math", "choice", "code", "instruction", "raw"})
RWKV_PROMPT_TEMPLATES = {
    "naive_cot": "User: {query}\n\nAssistant: <think",
    "naive_nocot": "User: {query}\n\nAssistant: <think></think>",
    "normal_cot": "User✿{query}✿\nBot✿<think",
    "normal_nocot": "User✿{query}✿\nBot✿<think></think>",
}

# CompletionRequest fields implemented by the vendored vllm-rwkv engine.
# ``model``, ``prompt`` and ``n`` are supplied by the client/profile contract,
# so putting them in [sampling] is an error rather than an override.
RWKV_COMPLETION_SAMPLING_FIELDS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "penalty_decay",
        "length_penalty",
        "seed",
        "stop",
        "stop_token_ids",
        "include_stop_str_in_output",
        "ignore_eos",
        "min_tokens",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
        "truncation_side",
        "use_beam_search",
        "allowed_token_ids",
        "prompt_logprobs",
        "logprob_token_ids",
        "bad_words",
    }
)


def _table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"RWKV evaluation profile requires a [{name}] table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str] | frozenset[str], section: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        rendered = ", ".join(unknown)
        raise ValueError(f"unsupported [{section}] field(s): {rendered}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class RWKVCompletionProfile:
    path: Path
    name: str
    model: str
    tasks: tuple[str, ...]
    adapter: str
    raw_input: bool
    prompt_mode: str
    prompt_template: str
    num_samples: int
    metric: str
    sampling: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "RWKVCompletionProfile":
        profile_path = Path(path).expanduser().resolve()
        if not profile_path.is_file():
            raise ValueError(f"RWKV evaluation profile not found: {profile_path}")
        with profile_path.open("rb") as file:
            config = tomllib.load(file)

        _reject_unknown(config, _TOP_LEVEL_FIELDS, "root")
        profile = _table(config, "profile")
        prompt = _table(config, "prompt")
        evaluation = _table(config, "evaluation")
        sampling = _table(config, "sampling")
        _reject_unknown(profile, _PROFILE_FIELDS, "profile")
        _reject_unknown(prompt, _PROMPT_FIELDS, "prompt")
        _reject_unknown(evaluation, _EVALUATION_FIELDS, "evaluation")
        _reject_unknown(sampling, RWKV_COMPLETION_SAMPLING_FIELDS, "sampling")

        name = str(profile.get("name") or "").strip()
        model = str(profile.get("model") or "").strip()
        adapter = str(profile.get("adapter") or "").strip()
        tasks_value = profile.get("tasks")
        if not name:
            raise ValueError("[profile].name must be a non-empty string")
        if not model:
            raise ValueError("[profile].model must be a non-empty string")
        if not adapter:
            raise ValueError("[profile].adapter must be a non-empty string")
        if adapter not in RWKV_PROFILE_ADAPTERS:
            allowed_adapters = ", ".join(sorted(RWKV_PROFILE_ADAPTERS))
            raise ValueError(f"[profile].adapter must be one of: {allowed_adapters}")
        if not isinstance(tasks_value, list) or not tasks_value:
            raise ValueError("[profile].tasks must be a non-empty array")
        tasks = tuple(str(task).strip() for task in tasks_value)
        if any(not task for task in tasks):
            raise ValueError("[profile].tasks cannot contain empty task names")
        if len(set(tasks)) != len(tasks):
            raise ValueError("[profile].tasks cannot contain duplicates")

        raw_input = profile.get("raw_input", True)
        if raw_input is not True:
            raise ValueError("[profile].raw_input must be true for RWKV completion profiles")

        prompt_mode = str(prompt.get("mode") or "").strip()
        prompt_template = prompt.get("template")
        if prompt_mode not in RWKV_PROMPT_TEMPLATES:
            allowed_modes = ", ".join(RWKV_PROMPT_TEMPLATES)
            raise ValueError(f"[prompt].mode must be one of: {allowed_modes}")
        if not isinstance(prompt_template, str):
            raise ValueError("[prompt].template must be a string")
        fields = [field_name for _, field_name, _, _ in Formatter().parse(prompt_template) if field_name]
        if fields != ["query"]:
            raise ValueError("[prompt].template must contain exactly one {query} field")
        if prompt_template != RWKV_PROMPT_TEMPLATES[prompt_mode]:
            raise ValueError(
                f"[prompt].template must be the raw RWKV wrapper for mode {prompt_mode!r}"
            )

        num_samples = _positive_int(evaluation.get("num_samples"), "[evaluation].num_samples")
        metric = str(evaluation.get("metric") or "").strip()
        if not metric:
            raise ValueError("[evaluation].metric must be a non-empty string")

        sampling = dict(sampling)
        sampling["max_tokens"] = _positive_int(sampling.get("max_tokens"), "[sampling].max_tokens")
        stop_token_ids = sampling.get("stop_token_ids")
        if stop_token_ids is not None and (
            not isinstance(stop_token_ids, list)
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in stop_token_ids)
        ):
            raise ValueError("[sampling].stop_token_ids must be an array of non-negative integers")
        if num_samples > 1 and float(sampling.get("temperature", 0.0)) <= 0:
            raise ValueError("[sampling].temperature must be greater than 0 when num_samples > 1")

        return cls(
            path=profile_path,
            name=name,
            model=model,
            tasks=tasks,
            adapter=adapter,
            raw_input=raw_input,
            prompt_mode=prompt_mode,
            prompt_template=prompt_template,
            num_samples=num_samples,
            metric=metric,
            sampling=sampling,
        )

    def render_prompt(self, query: str) -> str:
        return self.prompt_template.format(query=query)

    def completion_payload(self, *, served_model: str, prompt: str) -> dict[str, Any]:
        return {
            "model": served_model,
            "prompt": prompt,
            "n": self.num_samples,
            **self.sampling,
        }
