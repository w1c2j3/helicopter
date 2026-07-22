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
from functools import lru_cache
from pathlib import Path
from string import ascii_uppercase
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from lighteval.metrics.dynamic_metrics import ExprExtractionConfig, LatexExtractionConfig
from lighteval.metrics.metrics_sample import (
    AvgAtN,
    GPassAtK,
    JudgeLLM,
    SampleLevelComputation,
    SamplingMetric,
)
from lighteval.metrics.utils.extractive_match_utils import (
    extract_target_from_pred,
    get_extraction_regexes_inspect,
)
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    SampleLevelMetricGrouping,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.language import Language

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


def _gold_indices(doc: Doc) -> set[int]:
    value = doc.gold_index
    values = value if isinstance(value, (list, tuple)) else [value]
    indices: set[int] = set()
    for item in values:
        try:
            if item is not None:
                indices.add(int(item))
        except (TypeError, ValueError):
            continue
    return indices


def _is_generated_mcq(doc: Doc) -> bool:
    specific = getattr(doc, "specific", None)
    return isinstance(specific, Mapping) and bool(specific.get("helicopter_generated_mcq"))


def _choice_letter_score(doc: Doc, response: ModelResponse) -> float:
    """Score generated multiple-choice answers without log-likelihood calls."""

    gold_indices = {
        index
        for index in _gold_indices(doc)
        if 0 <= index < len(doc.choices)
    }
    if not gold_indices:
        return 0.0
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(doc.choices)])
    generated = getattr(response, "text_post_processed", None)
    if generated is None:
        generated = getattr(response, "text", None) or response.final_text
    if isinstance(generated, str):
        text = generated
    else:
        text = "\n".join(str(item or "") for item in generated)
    candidates: list[tuple[int, str]] = []
    patterns = (
        re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.IGNORECASE),
        re.compile(r"\\boxed\{\s*(?:\\(?:text|mathrm)\{\s*)?([A-Z])", re.IGNORECASE),
        re.compile(
            r"(?:final\s+)?(?:the\s+)?(?:correct\s+)?answer\s*"
            r"[*_]{0,3}\s*(?:is\s*)?:?\s*(?:option\s*)?[*_]{0,3}\s*"
            r"[\[(]?\s*[*_]{0,3}\s*([A-Z])\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"</think>\s*[*_]{0,3}\s*[\[(]?\s*([A-Z])\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:therefore|thus|hence|so)\b[^\n]{0,200}?"
            r"\b(?:is\s+provided\s+in|select(?:s|ed)?|choose|chosen|is)\s*"
            r"[*_]{0,3}\s*(?:option|choice)\s*[*_]{0,3}\s*([A-Z])\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:option|choice)\s*[*_]{0,3}\s*([A-Z])\b[^\n]{0,100}?"
            r"\b(?:is\s+)?(?:the\s+)?(?:correct|best|most\s+accurate)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:therefore|thus|hence|so),?\s*[*_]{0,3}\s*([A-Z])\s*[\).]",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:matches|corresponds\s+to)\s+(?:option|choice)\s*([A-Z])\b",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            label = match.group(1).upper()
            if label in allowed:
                candidates.append((match.start(), label))
    for match in re.finditer(r"\bis\s*[*_]{0,3}\s*([A-Z])\s*[\).:]", text, re.IGNORECASE):
        label = match.group(1).upper()
        if label in allowed:
            candidates.append((match.start(), label))

    prediction = max(candidates, default=(-1, ""), key=lambda item: item[0])[1]
    if not prediction:
        for line in reversed(text.splitlines()):
            match = re.fullmatch(r"\s*[\[(]?\s*([A-Z])\s*[\])]?\.?\s*", line, re.IGNORECASE)
            if match and match.group(1).upper() in allowed:
                prediction = match.group(1).upper()
                break

    if not prediction:
        specific = getattr(doc, "specific", None)
        choice_texts = (
            specific.get("helicopter_generated_mcq_choice_texts")
            if isinstance(specific, Mapping)
            else None
        )
        if isinstance(choice_texts, Mapping):
            possible = [text.strip()]
            possible.extend(
                match.group(1).strip()
                for match in re.finditer(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
            )
            nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
            if nonempty_lines:
                possible.append(nonempty_lines[-1])

            def normalized(value: Any) -> str:
                result = re.sub(r"\s+", " ", str(value)).strip().casefold()
                return re.sub(r"^(?:answer|option|choice)\s*[:：]\s*", "", result)

            normalized_choices = {
                str(label).upper(): normalized(value)
                for label, value in choice_texts.items()
                if str(label).upper() in allowed
            }
            for value in possible:
                candidate = normalized(value)
                matches = [
                    label
                    for label, choice in normalized_choices.items()
                    if choice and choice == candidate
                ]
                if len(matches) == 1:
                    prediction = matches[0]
                    break
    if not prediction:
        for line in reversed(text.splitlines()):
            match = re.match(r"\s*[*_]{0,3}[\[(]?\s*([A-Z])\s*[\]).:]\s+\S", line, re.IGNORECASE)
            if match and match.group(1).upper() in allowed:
                prediction = match.group(1).upper()
                break

    expected = {ascii_uppercase[index] for index in gold_indices}
    return 1.0 if prediction in expected else 0.0


def _single_prediction_score(sample_fn: Any, doc: Doc, response: ModelResponse) -> Any:
    """Run the original metric's per-prediction scorer."""

    score_doc = doc
    score_response = response
    if isinstance(sample_fn, SamplingMetric):
        scorer = sample_fn.compute_score
        # SamplingMetric.compute implementations normally preprocess references
        # and predictions before delegating to compute_score. The avg@k wrapper
        # calls that per-prediction scorer directly to persist rollout scores,
        # so reproduce the preprocessing contract here. LightEval task configs
        # in this checkout still set the legacy normalize_gold/normalize_pred
        # attributes, while SamplingMetric.preprocess uses the newer normalize
        # attribute, so honor both interfaces.
        def preprocess(value: Any, *, gold: bool) -> str:
            text = str(value)
            legacy = getattr(
                sample_fn,
                "normalize_gold" if gold else "normalize_pred",
                None,
            )
            if callable(legacy):
                if bool(getattr(sample_fn, "strip_strings", False)):
                    text = text.strip()
                return str(legacy(text))
            return str(sample_fn.preprocess(text))

        processed_golds = [
            preprocess(gold, gold=True)
            for gold in doc.get_golds()
        ]
        if processed_golds:
            score_doc = copy.deepcopy(doc)
            score_doc.choices = processed_golds
            score_doc.gold_index = (
                0 if len(processed_golds) == 1 else list(range(len(processed_golds)))
            )
        score_response = copy.deepcopy(response)
        score_response.text = [
            preprocess(prediction, gold=False)
            for prediction in response.final_text
        ]
        score_response.text_post_processed = None
    elif isinstance(sample_fn, SampleLevelComputation):
        scorer = sample_fn.compute
    else:
        scorer = getattr(sample_fn, "compute", sample_fn)
    if callable(scorer):
        # LightEval calls sample computations by keyword. Several native and
        # custom metrics declare these parameters in opposite orders, so
        # positional invocation is not a valid interface. Do not turn native
        # metric failures into choice scores: grouped metrics must preserve
        # their declared mapping contract, and evaluator defects should remain
        # visible instead of silently changing the benchmark semantics.
        value = scorer(doc=score_doc, model_response=score_response)
        if isinstance(value, Mapping):
            return dict(value)
        return float(value)
    return _choice_letter_score(doc, response)


@lru_cache(maxsize=2)
def _olympiad_bench_extraction_regexes(language: Language) -> Any:
    targets = (ExprExtractionConfig(), LatexExtractionConfig())
    return get_extraction_regexes_inspect(targets, language, len_choices=1)


def _olympiad_bench_score(doc: Doc, response: ModelResponse) -> float:
    """Apply the official OlympiadBench math-extraction scoring contract."""

    language = Language.CHINESE if "_zh_" in str(doc.task_name) else Language.ENGLISH
    regexes = _olympiad_bench_extraction_regexes(language)
    predictions = response.final_text
    prediction = str(predictions[0] if predictions else "")
    gold = "".join(str(item) for item in doc.get_golds())
    extracted_prediction = extract_target_from_pred(
        prediction,
        regexes,
        "first_match",
        "first_match",
        5,
    )
    extracted_gold = extract_target_from_pred(
        gold,
        regexes,
        "first_match",
        "first_match",
        5,
    )
    return float(extracted_prediction == extracted_gold)


class _RecordingAvgAtN(AvgAtN):
    """Avg@N that keeps each rollout's official score for DB eval rows."""

    def __init__(self, *, record_key: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.record_key = record_key

    @staticmethod
    def _record_scalar(value: Any) -> float:
        if isinstance(value, (bool, int, float, np.number)):
            return float(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return 0.0
            return float(np.mean([float(item) for item in value]))
        raise TypeError(f"unsupported rollout score value: {type(value).__name__}")

    @staticmethod
    def _aggregate_values(values: list[Any]) -> Any:
        if isinstance(values[0], (list, tuple)):
            if not values[0]:
                return []
            width = len(values[0])
            if any(not isinstance(value, (list, tuple)) or len(value) != width for value in values):
                raise ValueError("grouped metric returned inconsistent list lengths across rollouts")
            return [
                float(np.mean([float(value[index]) for value in values]))
                for index in range(width)
            ]
        return float(np.mean([float(value) for value in values]))

    @staticmethod
    def _zero_score(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: _RecordingAvgAtN._zero_score(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_RecordingAvgAtN._zero_score(item) for item in value]
        return 0.0

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> Any:
        rollout_scores = []
        processed = getattr(model_response, "text_post_processed", None)
        for index in range(int(self.n)):
            rollout = model_response[index]
            if processed is not None:
                rollout.text_post_processed = [
                    processed[index] if index < len(processed) else ""
                ]
            score = self.compute_score(doc, rollout)
            if processed is not None and not str(rollout.text_post_processed[0] or "").strip():
                score = self._zero_score(score)
            rollout_scores.append(score)
        recorded = getattr(model_response, "helicopter_rollout_scores", None)
        if not isinstance(recorded, dict):
            recorded = {}
            model_response.helicopter_rollout_scores = recorded

        if rollout_scores and all(isinstance(score, Mapping) for score in rollout_scores):
            aggregated: dict[str, Any] = {}
            for key in rollout_scores[0]:
                values = [score[key] for score in rollout_scores]
                recorded[str(key)] = [self._record_scalar(value) for value in values]
                aggregated[str(key)] = self._aggregate_values(values)
            return aggregated

        scores = [float(score) for score in rollout_scores]
        recorded[self.record_key] = scores
        return float(np.mean(scores))


def _is_g_pass(metric: Any) -> bool:
    sample_fn = getattr(metric, "sample_level_fn", None)
    names = getattr(metric, "metric_name", "")
    if isinstance(names, (tuple, list)):
        names = " ".join(str(item) for item in names)
    return isinstance(sample_fn, GPassAtK) or "g-pass@" in str(names).lower()


def _score_prediction(sample_fn: Any, doc: Doc, response: ModelResponse) -> Any:
    if "olympiad_bench:" in str(getattr(doc, "task_name", "")):
        return _olympiad_bench_score(doc, response)
    if _is_generated_mcq(doc):
        return _choice_letter_score(doc, response)
    return _single_prediction_score(sample_fn, doc, response)


def _generated_choice_avg_metric(*, k: int, name: str) -> SampleLevelMetric:
    """Represent a native log-likelihood MCQ as generated label accuracy."""

    return SampleLevelMetric(
        metric_name=name,
        sample_level_fn=_RecordingAvgAtN(
            record_key=name,
            n=k,
            sample_scoring_function=_choice_letter_score,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
        batched_compute=False,
    )


def _avg_metric(metric: Any, *, k: int, name: str) -> Any:
    sample_fn = getattr(metric, "sample_level_fn", None)
    if isinstance(sample_fn, JudgeLLM):
        deferred_name = f"deferred_judge_{name}"
        return SampleLevelMetric(
            metric_name=deferred_name,
            sample_level_fn=_RecordingAvgAtN(
                record_key=deferred_name,
                n=k,
                sample_scoring_function=lambda _doc, _response: 0.0,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
            batched_compute=False,
        )
    score_fn = lambda doc, response: _score_prediction(sample_fn, doc, response)
    metric_names = getattr(metric, "metric_name", None)
    if isinstance(metric_names, (list, tuple)):
        names = [str(item) for item in metric_names]
        corpus = getattr(metric, "corpus_level_fn", np.mean)
        if not isinstance(corpus, Mapping):
            corpus = dict.fromkeys(names, corpus)
        higher = getattr(metric, "higher_is_better", True)
        if not isinstance(higher, Mapping):
            higher = dict.fromkeys(names, bool(higher))
        return SampleLevelMetricGrouping(
            metric_name=names,
            sample_level_fn=_RecordingAvgAtN(
                record_key=name,
                n=k,
                sample_scoring_function=score_fn,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn={item: corpus[item] for item in names},
            higher_is_better={item: bool(higher[item]) for item in names},
            batched_compute=False,
        )
    return SampleLevelMetric(
        metric_name=name,
        sample_level_fn=_RecordingAvgAtN(
            record_key=name,
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


_TERMINAL_ANSWER_CUE_RE = re.compile(r"^[^\n]{1,40}[:：]\s*$")


def _choice_text(choice: Any, label: str) -> str:
    text = str(choice).strip()
    return re.sub(
        rf"^\s*(?:\({re.escape(label)}\)|{re.escape(label)}[.\):：])\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _query_contains_choices(query: str, choices: list[Any]) -> bool:
    """Detect native prompts that already serialize their answer choices."""

    labels = ascii_uppercase[: len(choices)]
    labelled = sum(
        bool(re.search(rf"(?m)^\s*(?:\({label}\)|{label}[.\):：])\s*\S", query, re.IGNORECASE))
        for label in labels
    )
    if labelled == len(choices):
        return True

    normalized_query = re.sub(r"\s+", " ", query).casefold()
    matched = 0
    for label, choice in zip(labels, choices):
        text = re.sub(r"\s+", " ", _choice_text(choice, label)).casefold()
        if len(text) >= 2 and text in normalized_query:
            matched += 1
    return matched == len(choices)


def _serialize_choices(query: str, choices: list[Any]) -> str:
    labels = ascii_uppercase[: len(choices)]
    option_lines = [
        f"{label}. {_choice_text(choice, label)}"
        for label, choice in zip(labels, choices)
    ]
    stripped = str(query).rstrip()
    lines = stripped.splitlines()
    cue = "Answer:"
    if lines and _TERMINAL_ANSWER_CUE_RE.fullmatch(lines[-1].strip()):
        cue = lines.pop().strip()
        stripped = "\n".join(lines).rstrip()
    sections = [section for section in (stripped, "\n".join(option_lines), cue) if section]
    return "\n".join(sections)


def _prepare_generated_mcq(doc: Doc, *, force: bool) -> Doc:
    """Preserve native LightEval choices when a task is evaluated by generation.

    Native log-likelihood tasks store candidate continuations in ``Doc.choices``
    and intentionally omit them from ``Doc.query``. Once the policy changes
    such a task to avg@k generation, those continuations must become labelled
    prompt options. Native generative MCQs are also marked so answer extraction
    is not accidentally applied to free-form tasks with multiple references.
    """

    choices = list(getattr(doc, "choices", ()) or ())
    if not 2 <= len(choices) <= len(ascii_uppercase):
        return doc
    gold_indices = _gold_indices(doc)
    single_gold = len(gold_indices) == 1
    if not force and not single_gold:
        return doc

    labels = list(ascii_uppercase[: len(choices)])
    choice_texts = {
        label: _choice_text(choice, label)
        for label, choice in zip(labels, choices)
    }
    normalized = [str(choice).strip().upper() for choice in choices]
    choices_are_labels = normalized == labels
    query = str(doc.query)
    if not choices_are_labels and not _query_contains_choices(query, choices):
        doc.query = _serialize_choices(query, choices)
    doc.choices = [f" {label}" for label in labels]
    specific = dict(getattr(doc, "specific", None) or {})
    specific["helicopter_generated_mcq"] = True
    specific["helicopter_generated_mcq_choice_texts"] = choice_texts
    doc.specific = specific
    return doc


def _wrap_prompt(
    prompt_function: Any,
    *,
    canonical_name: str,
    policy: Mapping[str, Any],
    force_choice_generation: bool,
) -> Any:
    def wrapped(line: dict[str, Any], task_name: str | None = None) -> Doc | None:
        # Several custom prompt functions branch on the catalog name.  Keep
        # that name for the inner formatter even though LightEval sees the
        # private alias.
        doc = prompt_function(line, canonical_name)
        if doc is None:
            return None
        doc = _prepare_generated_mcq(doc, force=force_choice_generation)
        # The raw endpoint adapter owns the final model prompt. Formatting it
        # here as well would produce nested ``User: ... Assistant:`` wrappers.
        if not os.environ.get("HELICOPTER_PROMPT_TEMPLATE"):
            doc.query = format_query(doc.query, canonical_name=canonical_name, policy=policy)
        return doc

    return wrapped


def _local_gpqa_prompt(line: dict[str, Any], task_name: str | None = None) -> Doc:
    """Adapt the authorized rwkv-skills GPQA cache to LightEval's native task contract."""

    labels = list(ascii_uppercase[:4])
    answer = str(line.get("answer", "")).strip().upper()
    if answer not in labels:
        raise ValueError(f"local GPQA row has invalid answer label: {answer!r}")
    instruction = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of ABCD. Think step by step before answering."
    )
    query = (
        f"{instruction}\n\n{str(line.get('question', '')).strip()}\n\n"
        + "\n".join(f"{label}) {str(line.get(label, '')).strip()}" for label in labels)
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=labels,
        gold_index=labels.index(answer),
        instruction=instruction,
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
    avg_k = int(policy["avg_k"])
    rollout_n = int(policy["rollout_n"])
    long_tasks = set(policy.get("long_rollout_tasks", []))

    cloned = _prefer_local_dataset(
        copy.deepcopy(config),
        canonical_name=canonical_name,
    )
    cloned.name = alias_task_name(canonical_name)
    cloned.full_name = f"{cloned.name}|0"
    force_choice_generation = any(
        getattr(metric, "category", None) == SamplingMethod.LOGPROBS
        for metric in cloned.metrics
    )
    cloned.prompt_function = _wrap_prompt(
        cloned.prompt_function,
        canonical_name=canonical_name,
        policy=policy,
        force_choice_generation=force_choice_generation,
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
    if force_choice_generation:
        cloned.metrics = (_generated_choice_avg_metric(k=avg_k, name=f"avg@{avg_k}"),)
        cloned.num_samples = [rollout_n]
        cloned.generation_size = int(
            policy["gpass_generation_size"]
            if canonical_name in long_tasks
            else policy["generation_size"]
        )
        return cloned
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
