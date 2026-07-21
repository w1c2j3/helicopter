"""LightEval task aliases for the TOML-driven G1h policy.

LightEval rejects custom task names that collide with its built-in registry.
The launcher therefore sends private ``g1h__...`` task names to LightEval and
keeps the original catalog names in the policy and scoreboard layer.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from lighteval.metrics.metrics_sample import (
    AvgAtN,
    GPassAtK,
    SampleLevelComputation,
    SamplingMetric,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod

from .g1h_config import alias_task_name, canonical_task_name, format_query, normalize_policy


POLICY_ENV = "HELICOPTER_LIGHTEEVAL_G1H_POLICY"
TASKS_ENV = "HELICOPTER_LIGHTEEVAL_TASKS"
_TASK_SPEC_SUFFIX_RE = re.compile(r"\|\d+$")


def _load_policy() -> dict[str, Any] | None:
    raw = os.environ.get(POLICY_ENV, "").strip()
    if not raw:
        return None
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid {POLICY_ENV}: {error}") from error
    if not isinstance(policy, dict):
        raise RuntimeError(f"{POLICY_ENV} must contain a JSON object")
    try:
        return normalize_policy(policy)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _selected_task_names(policy: Mapping[str, Any]) -> list[str]:
    configured = policy.get("selected_tasks")
    if isinstance(configured, list):
        return [str(item) for item in configured if str(item).strip()]
    raw = os.environ.get(TASKS_ENV, "")
    return [
        canonical_task_name(_TASK_SPEC_SUFFIX_RE.sub("", item.strip()))
        for item in raw.split(",")
        if item.strip()
    ]


def _gold_index(doc: Doc) -> int | None:
    value = doc.gold_index
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _choice_letter_score(doc: Doc, response: ModelResponse) -> float:
    """Score generated multiple-choice answers without log-likelihood calls."""

    gold_index = _gold_index(doc)
    if gold_index is None or gold_index < 0 or gold_index >= len(doc.choices):
        return 0.0
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)])
    text = "\n".join(str(item or "") for item in response.final_text)
    matches = re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text.upper())
    prediction = next((item for item in reversed(matches) if item in allowed), "")
    expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gold_index]
    return 1.0 if prediction == expected else 0.0


def _single_prediction_score(sample_fn: Any, doc: Doc, response: ModelResponse) -> float:
    """Run the original metric's per-prediction scorer."""

    if isinstance(sample_fn, SamplingMetric):
        scorer = sample_fn.compute_score
    elif isinstance(sample_fn, SampleLevelComputation):
        scorer = sample_fn.compute
    else:
        scorer = getattr(sample_fn, "compute", sample_fn)
    if callable(scorer):
        try:
            return float(scorer(doc, response))
        except (IndexError, KeyError, TypeError, ValueError):
            return _choice_letter_score(doc, response)
    return _choice_letter_score(doc, response)


def _is_g_pass(metric: Any) -> bool:
    sample_fn = getattr(metric, "sample_level_fn", None)
    names = getattr(metric, "metric_name", "")
    if isinstance(names, (tuple, list)):
        names = " ".join(str(item) for item in names)
    return isinstance(sample_fn, GPassAtK) or "g-pass@" in str(names).lower()


def _avg_metric(metric: Any, *, k: int, name: str) -> SampleLevelMetric:
    sample_fn = getattr(metric, "sample_level_fn", None)
    if getattr(metric, "category", None) is SamplingMethod.LOGPROBS:
        score_fn = _choice_letter_score
    else:
        score_fn = lambda doc, response: _single_prediction_score(sample_fn, doc, response)
    return SampleLevelMetric(
        metric_name=name,
        sample_level_fn=AvgAtN(
            n=k,
            sample_scoring_function=score_fn,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=bool(getattr(metric, "higher_is_better", True)),
        batched_compute=False,
    )


def _g_pass_metrics(
    metrics: Iterable[Any],
    *,
    policy: Mapping[str, Any],
) -> tuple[list[Any], int | None]:
    preserved: list[Any] = []
    configured_k = policy.get("gpass_k")
    configured_n = policy.get("gpass_n")
    effective_n: int | None = int(configured_n) if configured_n is not None else None
    for metric in metrics:
        if not _is_g_pass(metric):
            continue
        cloned = copy.deepcopy(metric)
        sample_fn = getattr(cloned, "sample_level_fn", None)
        if isinstance(sample_fn, GPassAtK):
            if configured_k is not None:
                sample_fn.k = [int(configured_k)]
            if configured_n is not None:
                sample_fn.n = int(configured_n)
            if effective_n is None and sample_fn.n is not None:
                effective_n = int(sample_fn.n)
        preserved.append(cloned)
    return preserved, effective_n


def _wrap_prompt(
    prompt_function: Any,
    *,
    canonical_name: str,
    policy: Mapping[str, Any],
) -> Any:
    def wrapped(line: dict[str, Any], task_name: str | None = None) -> Doc | None:
        # Several custom prompt functions branch on the catalog name.  Keep
        # that name for the inner formatter even though LightEval sees the
        # private alias.
        doc = prompt_function(line, canonical_name)
        if doc is None:
            return None
        doc.query = format_query(doc.query, canonical_name=canonical_name, policy=policy)
        return doc

    return wrapped


def _policy_config(
    config: LightevalTaskConfig,
    *,
    canonical_name: str,
    policy: Mapping[str, Any],
) -> LightevalTaskConfig:
    avg_k = int(policy["avg_k"])
    rollout_n = int(policy["rollout_n"])
    long_tasks = set(policy.get("long_rollout_tasks", []))

    cloned = copy.deepcopy(config)
    cloned.name = alias_task_name(canonical_name)
    cloned.full_name = f"{cloned.name}|0"
    cloned.prompt_function = _wrap_prompt(
        config.prompt_function,
        canonical_name=canonical_name,
        policy=policy,
    )

    # Explicit zero-shot means no dev/train examples and an explicit |0 in
    # the LightEval task selector.  This also overrides built-ins such as
    # CEval/MMLU that carry a few-shot split for other launchers.
    if policy.get("zero_shot", True):
        cloned.few_shots_split = None
        cloned.few_shots_select = None
        cloned.num_fewshots = 0

    gpass_metrics, gpass_n = _g_pass_metrics(cloned.metrics, policy=policy)
    if gpass_metrics:
        cloned.metrics = tuple(gpass_metrics)
        if gpass_n is not None:
            cloned.num_samples = [gpass_n]
        cloned.generation_size = int(policy["gpass_generation_size"])
        return cloned

    metrics = list(cloned.metrics)
    if not metrics:
        raise RuntimeError(f"g1h policy cannot configure task {canonical_name!r}: no metrics")
    names = ["avg@%d" % avg_k] if len(metrics) == 1 else [f"avg@{avg_k}_{i}" for i in range(len(metrics))]
    cloned.metrics = tuple(
        _avg_metric(metric, k=avg_k, name=name)
        for metric, name in zip(metrics, names)
    )
    cloned.num_samples = [rollout_n]
    cloned.generation_size = int(
        policy["gpass_generation_size"]
        if canonical_name in long_tasks
        else policy["generation_size"]
    )
    return cloned


def apply_g1h_policy(custom_tasks: Iterable[LightevalTaskConfig]) -> list[LightevalTaskConfig]:
    """Add uniquely named configured aliases for the selected tasks only."""

    policy = _load_policy()
    original_tasks = list(custom_tasks)
    if policy is None:
        return original_tasks

    selected = _selected_task_names(policy)
    if not selected:
        # Do not expand tens of thousands of built-in tasks when the module is
        # imported for ordinary task inspection.
        return original_tasks

    builtin = Registry.load_all_task_configs(custom_tasks=None, load_multilingual=True)
    available: dict[str, LightevalTaskConfig] = dict(builtin)
    available.update({config.name: config for config in original_tasks})

    aliases: list[LightevalTaskConfig] = []
    for canonical_name in selected:
        config = available.get(canonical_name)
        if config is None:
            raise RuntimeError(f"g1h policy selected task {canonical_name!r}, but LightEval has no such task")
        aliases.append(_policy_config(config, canonical_name=canonical_name, policy=policy))
    return [*original_tasks, *aliases]


__all__ = ["apply_g1h_policy"]
