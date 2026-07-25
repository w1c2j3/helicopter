"""LightEval task aliases for the TOML-driven G1h policy.

The policy changes sampling only through LightEval's own metric classes.  The
avg branch uses the official :class:`AvgAtN` and keeps the benchmark-specific
single-completion scorer.  Log-probability multiple-choice tasks receive a
generative choice scorer only in that avg branch; their native LOGPROBS branch
is left unchanged.
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

import numpy as np

from lighteval.metrics.metrics_sample import (
    AvgAtN,
    GPassAtK,
    JudgeLLM,
    MajAtN,
    PassAtK,
    SampleLevelComputation,
)
from lighteval.metrics.avg_at_n import build_avg_at_n_metric
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


class _CanonicalAnswerScorer(SampleLevelComputation):
    """Score one rollout with the same adapter used by the DB eval rows."""

    def __init__(self, *, domain: str, request_format: str):
        self.domain = str(domain or "").strip().lower()
        self.request_format = str(request_format or "").strip().lower()

    def compute(self, doc: Doc, model_response: Any, **kwargs: Any) -> float:
        del kwargs
        predictions = list(getattr(model_response, "final_text", []) or [])
        if not any(str(item or "").strip() for item in predictions):
            predictions = list(getattr(model_response, "text", []) or [])
        if not predictions:
            return 0.0
        prediction = str(predictions[0] or "")

        from helicopter_cli.lighteval_answer_adapters import answers_match

        format_name = self.request_format
        if format_name in {"choice", "multiple_choice", "multichoice", "mmlu"}:
            choices = list(getattr(doc, "choices", None) or [])
            gold_indices = getattr(doc, "gold_index", [])
            if not isinstance(gold_indices, (list, tuple, set)):
                gold_indices = [gold_indices]
            golds = [
                ascii_uppercase[int(index)]
                for index in gold_indices
                if isinstance(index, int) and 0 <= int(index) < len(ascii_uppercase)
            ]
        else:
            golds = list(doc.get_golds())

        for gold in golds:
            matched = answers_match(
                prediction,
                str(gold),
                domain=self.domain,
                request_format=self.request_format,
            )
            if matched is True:
                return 1.0
        return 0.0


def _avg_metric(
    metric: Any,
    *,
    k: int,
    name: str,
    domain: str | None = None,
    request_format: str | None = None,
) -> Any:
    """Build a real LightEval ``AvgAtN`` metric without changing native mode."""

    sample_fn = getattr(metric, "sample_level_fn", None)

    if sample_fn is None or not type(sample_fn).__module__.startswith("lighteval."):
        return _official_fallback_metric(metric)

    domain_name = str(domain or "").strip().lower()
    format_name = str(request_format or "").strip().lower()
    if domain_name == "math" or format_name in {"choice", "multiple_choice", "multichoice", "mmlu"}:
        return SampleLevelMetric(
            metric_name=name,
            sample_level_fn=AvgAtN(
                n=int(k),
                sample_scoring_function=_CanonicalAnswerScorer(
                    domain=domain_name,
                    request_format=format_name,
                ),
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )
    return build_avg_at_n_metric(metric, k=int(k), name=name)


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


def _request_policy_from_environment() -> Mapping[str, Any]:
    """Read the task request policy shared by prompt and score adapters."""

    raw = os.environ.get("HELICOPTER_LIGHTEEVAL_TASK_REQUEST_POLICY", "")
    try:
        payload = json.loads(raw)
        tasks = payload.get("tasks", {})
        if isinstance(tasks, Mapping) and len(tasks) == 1:
            entry = next(iter(tasks.values()))
            if isinstance(entry, Mapping):
                return entry
    except (TypeError, ValueError, StopIteration):
        pass
    return {}


def _normalize_doc_references(
    doc: Doc,
    *,
    domain: str | None,
    request_format: str | None,
) -> None:
    """Apply the same answer adapter to every reference consumed by metrics.

    ``Doc.choices`` is normally the gold value for free-answer tasks. For
    multiple-choice tasks it is the option table, so rewriting all options
    would break LOGPROBS and native choice metrics; only single-gold choice
    documents are adapted. Explicit reference lists used by judge/custom
    tasks receive the same treatment.
    """

    from helicopter_cli.lighteval_answer_adapters import adapt_answer

    domain_name = str(domain or "").strip().lower()
    format_name = str(request_format or "").strip().lower()
    choices = getattr(doc, "choices", None)
    if isinstance(choices, list):
        is_math = domain_name == "math" or format_name in {
            "math",
            "math_boxed",
            "math_choice",
            "formal_proof",
        }
        is_code = domain_name == "coding" or format_name in {
            "code",
            "code_completion",
            "diff_patch",
            "generic_code",
            "python_function",
            "python_program",
            "python_snippet",
        }
        # Multiple-choice options are not gold answers and must remain raw.
        if len(choices) == 1 or is_math or is_code:
            doc.choices = [
                adapt_answer(
                    str(choice),
                    domain=domain,
                    request_format=request_format,
                    prompt=str(getattr(doc, "query", "") or ""),
                )
                if isinstance(choice, (str, int, float))
                else choice
                for choice in choices
            ]

    specific = getattr(doc, "specific", None)
    if not isinstance(specific, dict):
        return
    reference_keys = {
        "answer",
        "expected_answer",
        "reference_answer",
        "reference",
        "reference_answers",
        "references",
        "solution",
        "target",
        "gold_answer",
        "gold_patch",
        "reference_plan",
        "reference_plans",
    }
    for key in reference_keys:
        value = specific.get(key)
        if isinstance(value, (str, int, float)):
            specific[key] = adapt_answer(
                str(value),
                domain=domain,
                request_format=request_format,
                prompt=str(getattr(doc, "query", "") or ""),
            )
        elif isinstance(value, list):
            specific[key] = [
                adapt_answer(
                    str(item),
                    domain=domain,
                    request_format=request_format,
                    prompt=str(getattr(doc, "query", "") or ""),
                )
                if isinstance(item, (str, int, float))
                else item
                for item in value
            ]


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
            if not os.environ.get("HELICOPTER_PROMPT_TEMPLATE"):
                doc.query = format_query(doc.query, canonical_name=canonical_name, policy=policy)
            request_policy = _request_policy_from_environment()
            request_format = (
                str(request_policy.get("format")).strip().lower()
                if request_policy.get("format")
                else None
            )
            _normalize_doc_references(
                doc,
                domain=policy.get("domain") or request_policy.get("domain"),
                request_format=request_format,
            )
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
    selected_metric = str(effective_policy.get("metric", "avg")).strip().lower()

    # GPass remains an independent native branch.  When the TOML selects avg,
    # it goes through the same AvgAtN conversion as every other sampling
    # metric, so gpass-specific TOML controls cannot silently override metric.
    if selected_metric == "native":
        gpass_metrics, gpass_n = _g_pass_metrics(cloned.metrics, policy=effective_policy)
        if gpass_metrics:
            cloned.metrics = tuple(gpass_metrics)
            if gpass_n is not None:
                cloned.num_samples = [gpass_n]
            cloned.generation_size = int(
                effective_policy.get("gpass_generation_size", policy["gpass_generation_size"])
            )
            return cloned

    # The benchmark TOML decides whether this task uses avg or its task-native
    # metric.  The native branch below is deliberately untouched; only the
    # avg branch converts the selected native scorer to a per-completion
    # scorer wrapped by LightEval's official AvgAtN.
    if selected_metric == "native":
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
    request_policy = _request_policy_from_environment()
    request_domain = str(request_policy.get("domain") or "").strip().lower()
    request_format = str(request_policy.get("format") or "").strip().lower()
    cloned.metrics = tuple(
        _avg_metric(
            metric,
            k=avg_k,
            name=name,
            domain=request_domain,
            request_format=request_format,
        )
        for metric, name in zip(metrics, names)
    )

    # Every supported avg metric is represented by AvgAtN, so LightEval's
    # request builder asks the model for exactly avg_k completions.
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
