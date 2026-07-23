"""LightEval task aliases for the TOML-driven G1h policy.

The policy changes sampling only through LightEval's own metric classes.  In
particular, ordinary avg@k uses the official :class:`AvgAtN` with the task's
native sample scorer as its ``sample_scoring_function``.  It never replaces a
benchmark scorer with a project-wide parser, judge, or regular expression.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from lighteval.metrics.metrics_sample import (
    AvgAtN,
    GPassAtK,
    JudgeLLM,
    MajAtN,
    PassAtK,
    SampleLevelComputation,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod

try:
    from langdetect import DetectorFactory
except ImportError:  # pragma: no cover - LightEval declares it for IFEval
    DetectorFactory = None
else:
    DetectorFactory.seed = 0

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


def _is_g_pass(metric: Any) -> bool:
    sample_fn = getattr(metric, "sample_level_fn", None)
    names = getattr(metric, "metric_name", "")
    if isinstance(names, (tuple, list)):
        names = " ".join(str(item) for item in names)
    return isinstance(sample_fn, GPassAtK) or "g-pass@" in str(names).lower()


def _avg_metric(metric: Any, *, k: int, name: str) -> Any:
    """Configure one metric using only LightEval's native scorer contract.

    ``AvgAtN`` itself is reused and configured in place on a copy.  Otherwise,
    a scalar generative native scorer is passed directly to a new official
    ``AvgAtN``.  Judge, pass-at-k, grouped, batched, and non-generative
    LightEval metrics are preserved because their contracts are not an
    ordinary mean of independent scalar rollouts. Project-defined scorers are
    replaced by an official LightEval metric and are never called.
    """

    sample_fn = getattr(metric, "sample_level_fn", None)
    metric_names = getattr(metric, "metric_name", None)

    if sample_fn is None or not type(sample_fn).__module__.startswith("lighteval."):
        return _official_fallback_metric(metric)

    if isinstance(sample_fn, AvgAtN):
        cloned = copy.deepcopy(metric)
        cloned.sample_level_fn.n = int(k)
        cloned.metric_name = name
        return cloned

    if (
        not isinstance(sample_fn, SampleLevelComputation)
        or isinstance(sample_fn, (JudgeLLM, MajAtN, PassAtK, GPassAtK))
        or isinstance(metric_names, (list, tuple))
        or getattr(metric, "batched_compute", False)
        or getattr(metric, "category", None) != SamplingMethod.GENERATIVE
    ):
        return copy.deepcopy(metric)

    return SampleLevelMetric(
        metric_name=name,
        sample_level_fn=AvgAtN(
            n=int(k),
            sample_scoring_function=copy.deepcopy(sample_fn),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=copy.deepcopy(getattr(metric, "corpus_level_fn")),
        higher_is_better=bool(getattr(metric, "higher_is_better", True)),
        batched_compute=False,
    )


def _official_fallback_metric(metric: Any) -> Any:
    """Use a LightEval metric when a custom task has no native scorer.

    A project-defined ``SampleLevelComputation`` cannot be made native by
    wrapping it in ``AvgAtN``. Keep such tasks on a LightEval-owned metric
    instead of allowing the old custom scorer to execute in a rollout.
    """

    del metric
    return copy.deepcopy(Metrics.exact_match.value)


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


def _configure_native_metrics(
    metrics: Iterable[Any],
    *,
    policy: Mapping[str, Any],
) -> list[Any]:
    """Apply TOML k/n controls while keeping each official scorer intact."""

    configured_pass_k = policy.get("pass_k")
    configured_pass_n = policy.get("pass_n")
    configured_native_n = policy.get("native_n")
    configured_gpass_k = policy.get("gpass_k")
    configured_gpass_n = policy.get("gpass_n")
    configured_rollout_n = policy.get("rollout_n")
    configured: list[Any] = []
    for metric in metrics:
        cloned = copy.deepcopy(metric)
        sample_fn = getattr(cloned, "sample_level_fn", None)
        if isinstance(sample_fn, PassAtK):
            if configured_pass_k is not None:
                sample_fn.k = int(configured_pass_k)
            if configured_pass_n is not None:
                sample_fn.n = int(configured_pass_n)
            elif configured_rollout_n is not None and sample_fn.n is None:
                sample_fn.n = int(configured_rollout_n)
        elif isinstance(sample_fn, GPassAtK):
            if configured_gpass_k is not None:
                sample_fn.k = [int(configured_gpass_k)]
            if configured_gpass_n is not None:
                sample_fn.n = int(configured_gpass_n)
            elif configured_rollout_n is not None and sample_fn.n is None:
                sample_fn.n = int(configured_rollout_n)
        elif isinstance(sample_fn, MajAtN):
            if configured_native_n is not None:
                sample_fn.n = int(configured_native_n)
            elif configured_rollout_n is not None:
                sample_fn.n = int(configured_rollout_n)
        configured.append(cloned)
    return configured


def _metrics_for_avg(metrics: Iterable[Any]) -> list[Any]:
    """Prefer a task-declared native AvgAtN over a duplicate base metric."""

    values = list(metrics)
    native_avg = [
        metric
        for metric in values
        if isinstance(getattr(metric, "sample_level_fn", None), AvgAtN)
    ]
    return native_avg or values


def _wrap_prompt(
    prompt_function: Any,
    *,
    canonical_name: str,
    policy: Mapping[str, Any],
) -> Any:
    def wrapped(line: dict[str, Any], task_name: str | None = None) -> Doc | None:
        # Custom prompt functions may branch on the catalog name. Keep that
        # name for formatting even though LightEval receives a private alias.
        formatted = prompt_function(line, canonical_name)
        if formatted is None:
            return None

        def prepare(doc: Doc) -> Doc:
            # Do not rewrite Doc.choices or mark generated MCQs: those fields
            # belong to the benchmark's native scorer and sampling method.
            if not os.environ.get("HELICOPTER_PROMPT_TEMPLATE"):
                doc.query = format_query(doc.query, canonical_name=canonical_name, policy=policy)
            return doc

        if isinstance(formatted, list):
            return [prepare(doc) for doc in formatted]
        return prepare(formatted)

    return wrapped


def _local_gpqa_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    """Adapt the authorized local GPQA row without changing its scorer."""

    labels = list(ascii_uppercase[:4])
    answer = str(line.get("answer", "")).strip().upper()
    if answer not in labels:
        raise ValueError(f"local GPQA row has invalid answer label: {answer!r}")
    query = str(line.get("question", "")).strip()
    query += "\n" + "\n".join(
        f"{label}. {str(line.get(label, '')).strip()}" for label in labels
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=labels,
        gold_index=labels.index(answer),
        instruction=None,
    )


def _prefer_local_dataset(
    config: LightevalTaskConfig,
    *,
    canonical_name: str,
) -> LightevalTaskConfig:
    root_value = os.environ.get("DATASETS_PATH", "").strip()
    if not root_value:
        return config
    root = Path(root_value)
    local: tuple[Path, str] | None = None
    if canonical_name == "math_500":
        local = (root / "cache/math_500/math_500_test.jsonl", "test")
    elif canonical_name == "ifbench_test":
        local = (root / "cache/ifbench/IFBench_test.jsonl", "train")
    elif canonical_name.startswith("mmlu:"):
        subject = canonical_name.split(":", 1)[1]
        local = (root / "cache/lighteval_mmlu" / f"{subject}.jsonl", "test")
    elif canonical_name.startswith("gpqa:"):
        variant = canonical_name.split(":", 1)[1]
        split = "main" if variant == "mc" else variant
        if split in {"main", "diamond", "extended"}:
            local = (root / "gpqa" / f"{split}.jsonl", "train")
    if local is None or not local[0].is_file():
        return config

    path, split = local
    config.hf_repo = "json"
    config.hf_subset = "default"
    config.hf_data_files = {split: str(path)}
    config.hf_avail_splits = (split,)
    config.evaluation_splits = (split,)
    config.few_shots_split = None
    if canonical_name.startswith("gpqa:"):
        config.prompt_function = _local_gpqa_prompt
    return config


def _policy_config(
    config: LightevalTaskConfig,
    *,
    canonical_name: str,
    policy: Mapping[str, Any],
) -> LightevalTaskConfig:
    task_evaluations = policy.get("task_evaluations", {})
    task_evaluation: Mapping[str, Any] = {}
    if isinstance(task_evaluations, Mapping):
        candidate = task_evaluations.get(canonical_name)
        if candidate is None:
            parent_matches = [
                (name, value)
                for name, value in task_evaluations.items()
                if canonical_name.startswith(f"{name}:")
            ]
            if parent_matches:
                candidate = max(parent_matches, key=lambda item: len(item[0]))[1]
        if isinstance(candidate, Mapping):
            task_evaluation = candidate
    effective_policy: dict[str, Any] = {**policy, **task_evaluation}
    avg_k = int(effective_policy["avg_k"])
    long_tasks = set(effective_policy.get("long_rollout_tasks", []))

    cloned = _prefer_local_dataset(
        copy.deepcopy(config),
        canonical_name=canonical_name,
    )
    cloned.name = alias_task_name(canonical_name)
    cloned.full_name = f"{cloned.name}|0"
    cloned.prompt_function = _wrap_prompt(
        cloned.prompt_function,
        canonical_name=canonical_name,
        policy=policy,
    )

    if policy.get("zero_shot", True):
        cloned.few_shots_split = None
        cloned.few_shots_select = None
        cloned.num_fewshots = 0

    cloned.metrics = tuple(_configure_native_metrics(cloned.metrics, policy=effective_policy))
    gpass_metrics, gpass_n = _g_pass_metrics(cloned.metrics, policy=effective_policy)
    if gpass_metrics:
        cloned.metrics = tuple(gpass_metrics)
        if gpass_n is not None:
            cloned.num_samples = [gpass_n]
        cloned.generation_size = int(
            effective_policy.get("gpass_generation_size", policy["gpass_generation_size"])
        )
        return cloned

    # Native benchmark contracts stay native.  In particular, do not wrap
    # pass@k, maj@N, judge, or another official LightEval metric in AvgAtN just
    # because the surrounding launcher has an avg policy section.  The
    # benchmark TOML decides whether this task uses avg or its task-native
    # metric, and the native metric's own k/n controls were applied above.
    if str(effective_policy.get("metric", "avg")).strip().lower() == "native":
        native_n = effective_policy.get("native_n")
        rollout_n = effective_policy.get("rollout_n")
        if native_n is not None:
            cloned.num_samples = [int(native_n)]
        elif rollout_n is not None:
            cloned.num_samples = [int(rollout_n)]
        if any(
            getattr(metric, "category", None) == SamplingMethod.GENERATIVE
            for metric in cloned.metrics
        ):
            cloned.generation_size = int(
                effective_policy.get("gpass_generation_size", policy["gpass_generation_size"])
                if canonical_name in long_tasks
                else effective_policy.get("generation_size", policy["generation_size"])
            )
        return cloned

    metrics = _metrics_for_avg(cloned.metrics)
    if not metrics:
        raise RuntimeError(f"g1h policy cannot configure task {canonical_name!r}: no metrics")
    names = [
        f"avg@{avg_k}" if len(metrics) == 1 else f"avg@{avg_k}_{index}"
        for index in range(len(metrics))
    ]
    cloned.metrics = tuple(
        _avg_metric(metric, k=avg_k, name=name)
        for metric, name in zip(metrics, names)
    )

    # Only official AvgAtN needs avg_k responses. Unsupported native contracts
    # keep the task's original num_samples and metric implementation.
    uses_avg_at_n = any(
        isinstance(getattr(metric, "sample_level_fn", None), AvgAtN)
        for metric in cloned.metrics
    )
    if uses_avg_at_n:
        cloned.num_samples = [avg_k]
    elif effective_policy.get("rollout_n") is not None:
        cloned.num_samples = [int(effective_policy["rollout_n"])]
    if any(
        getattr(metric, "category", None) == SamplingMethod.GENERATIVE
        for metric in cloned.metrics
    ):
        cloned.generation_size = int(
            effective_policy.get("gpass_generation_size", policy["gpass_generation_size"])
            if canonical_name in long_tasks
            else effective_policy.get("generation_size", policy["generation_size"])
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
        return original_tasks

    builtin = Registry.load_all_task_configs(custom_tasks=None, load_multilingual=True)
    available: dict[str, LightevalTaskConfig] = dict(builtin)
    # A custom task file is allowed to add benchmarks absent from LightEval,
    # but it must not shadow an official task with the same name. Otherwise a
    # native benchmark scorer would silently become a project scorer before
    # the policy even gets a chance to configure it.
    for config in original_tasks:
        available.setdefault(config.name, config)

    aliases: list[LightevalTaskConfig] = []
    for canonical_name in selected:
        config = available.get(canonical_name)
        if config is not None:
            aliases.append(_policy_config(config, canonical_name=canonical_name, policy=policy))
            continue

        family_prefix = f"{canonical_name}:"
        family = sorted(
            (
                (name, child)
                for name, child in available.items()
                if name.startswith(family_prefix)
            ),
            key=lambda item: item[0],
        )
        if not family:
            raise RuntimeError(
                f"g1h policy selected task {canonical_name!r}, but LightEval has no such task"
            )
        aliases.extend(
            _policy_config(child, canonical_name=name, policy=policy)
            for name, child in family
        )
    return [*original_tasks, *aliases]


__all__ = ["apply_g1h_policy", "_avg_metric", "_metrics_for_avg", "_policy_config"]
