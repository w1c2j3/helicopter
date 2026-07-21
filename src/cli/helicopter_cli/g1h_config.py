"""Small, dependency-free helpers for the TOML-driven G1h policy.

The LightEval registry deliberately rejects custom tasks whose names collide
with built-in tasks.  The launcher therefore keeps the catalog names as the
public identity and uses a private ``g1h__`` alias only inside a configured
run.  This module contains the shared parsing and selection rules used by the
launcher and by the spawned custom-task module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


G1H_TASK_PREFIX = "g1h__"
_FEWSHOT_SUFFIX_RE = re.compile(r"\|\d+$")


def _positive_int(policy: Mapping[str, Any], key: str, default: int) -> int:
    value = policy.get(key, default)
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"[lighteval.g1h].{key} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"[lighteval.g1h].{key} must be a positive integer")
    return result


def _string_list(policy: Mapping[str, Any], key: str) -> list[str]:
    value = policy.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"[lighteval.g1h].{key} must be a TOML array")
    return [str(item) for item in value if str(item).strip()]


def normalize_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the public TOML policy."""

    result = dict(policy)
    result["metric"] = str(result.get("metric", "avg")).strip().lower()
    if result["metric"] != "avg":
        raise ValueError("[lighteval.g1h].metric must be 'avg'")
    result["prompt_style"] = str(result.get("prompt_style", "naive")).strip().lower()
    if result["prompt_style"] not in {"naive", "normal"}:
        raise ValueError("[lighteval.g1h].prompt_style must be 'naive' or 'normal'")

    result["zero_shot"] = bool(result.get("zero_shot", True))
    result["avg_k"] = _positive_int(result, "avg_k", 8)
    result["rollout_n"] = _positive_int(result, "rollout_n", result["avg_k"])
    if result["rollout_n"] != result["avg_k"]:
        raise ValueError("[lighteval.g1h] requires rollout_n == avg_k for ordinary avg@k")

    result["generation_size"] = _positive_int(result, "generation_size", 4096)
    result["gpass_generation_size"] = _positive_int(result, "gpass_generation_size", 8192)
    gpass_k = result.get("gpass_k")
    gpass_n = result.get("gpass_n")
    if gpass_k is not None:
        result["gpass_k"] = _positive_int(result, "gpass_k", result["avg_k"])
    if gpass_n is not None:
        result["gpass_n"] = _positive_int(result, "gpass_n", result["gpass_k"] if gpass_k else result["avg_k"])
    if result.get("gpass_k") is not None and result.get("gpass_n") is not None:
        if result["gpass_k"] > result["gpass_n"]:
            raise ValueError("[lighteval.g1h].gpass_k cannot exceed gpass_n")

    result["long_rollout_tasks"] = _string_list(result, "long_rollout_tasks")
    result["no_cot_tasks"] = _string_list(result, "no_cot_tasks")
    result["variant_selection"] = str(result.get("variant_selection", "avg_then_gpass")).strip().lower()
    if result["variant_selection"] not in {"all", "avg_then_gpass"}:
        raise ValueError("[lighteval.g1h].variant_selection must be 'all' or 'avg_then_gpass'")

    result.setdefault("naive_cot_template", "User: {query}\nAssistant: <think>")
    result.setdefault("naive_nocot_template", "User: {query}\nAssistant:")
    result.setdefault("normal_cot_template", "User✿{query}✿\nBot✿<think>")
    result.setdefault("normal_nocot_template", "User✿{query}✿\nBot✿<think></think>")
    return result


def task_name_from_spec(spec: str) -> str:
    """Return the catalog task name from ``task|fewshot[@params]``."""

    name = _FEWSHOT_SUFFIX_RE.sub("", str(spec).strip())
    return name.split("@", 1)[0].strip()


def _split_spec(spec: str, *, zero_shot: bool) -> tuple[str, str]:
    raw = str(spec).strip()
    if not raw:
        return "", "0"
    task, separator, fewshot = raw.partition("|")
    task = task.split("@", 1)[0].strip()
    if zero_shot or not separator:
        fewshot = "0"
    return task, fewshot


def _variant_group(name: str) -> tuple[str, str] | None:
    for suffix, variant in (("_avg", "avg"), ("_gpassk", "gpass")):
        if name.endswith(suffix):
            return name[: -len(suffix)], variant
    return None


def select_task_specs(specs: Iterable[str], policy: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Select one canonical task variant and make zero-shot explicit.

    For a catalog containing ``aime24``, ``aime24_avg`` and
    ``aime24_gpassk``, the default policy selects ``aime24_avg``.  If no
    ordinary avg variant exists, it selects the G-pass variant.  This avoids
    measuring the same benchmark three times under different metrics.
    """

    normalized = normalize_policy(policy)
    parsed = [_split_spec(spec, zero_shot=normalized["zero_shot"]) for spec in specs]
    parsed = [(name, fewshot) for name, fewshot in parsed if name]
    if normalized["variant_selection"] == "all":
        return list(dict.fromkeys(parsed))

    grouped: dict[str, list[tuple[str, str, str | None]]] = {}
    order: list[str] = []
    for name, fewshot in parsed:
        variant_info = _variant_group(name)
        if variant_info is None:
            key = name
            variant = None
        else:
            key, variant = variant_info
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((name, fewshot, variant))

    selected: list[tuple[str, str]] = []
    for key in order:
        candidates = grouped[key]
        avg = next((item for item in candidates if item[2] == "avg"), None)
        gpass = next((item for item in candidates if item[2] == "gpass"), None)
        plain = next((item for item in candidates if item[2] is None), None)
        choice = avg or gpass or plain
        if choice is not None:
            selected.append((choice[0], choice[1]))
    return selected


def alias_task_name(canonical_name: str) -> str:
    return f"{G1H_TASK_PREFIX}{canonical_name}"


def alias_task_specs(specs: Iterable[tuple[str, str]], policy: Mapping[str, Any]) -> list[str]:
    normalized = normalize_policy(policy)
    return [
        f"{alias_task_name(name)}|{fewshot if not normalized['zero_shot'] else '0'}"
        for name, fewshot in specs
    ]


def canonical_task_name(alias_or_name: str) -> str:
    name = task_name_from_spec(alias_or_name)
    if name.startswith(G1H_TASK_PREFIX):
        return name[len(G1H_TASK_PREFIX) :]
    return name


def task_uses_cot(canonical_name: str, policy: Mapping[str, Any]) -> bool:
    normalized = normalize_policy(policy)
    return not any(
        canonical_name == item or canonical_name.startswith(f"{item}:")
        for item in normalized["no_cot_tasks"]
    )


def format_query(query: str, *, canonical_name: str, policy: Mapping[str, Any]) -> str:
    normalized = normalize_policy(policy)
    cot = task_uses_cot(canonical_name, normalized)
    style = normalized["prompt_style"]
    key = (
        "normal_cot_template"
        if style == "normal" and cot
        else "normal_nocot_template"
        if style == "normal"
        else "naive_cot_template"
        if cot
        else "naive_nocot_template"
    )
    template = str(normalized[key])
    try:
        return template.format(query=str(query))
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid prompt template {key}: {error}") from error


__all__ = [
    "G1H_TASK_PREFIX",
    "alias_task_name",
    "alias_task_specs",
    "canonical_task_name",
    "format_query",
    "normalize_policy",
    "select_task_specs",
    "task_name_from_spec",
]
