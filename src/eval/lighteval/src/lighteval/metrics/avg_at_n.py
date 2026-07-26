"""Helpers for configuring real generative ``avg@k`` metrics.

The helper in this module deliberately keeps the task's native metric intact
for the native branch.  An avg metric is a separate generative metric: the
model must return ``k`` completions and the single-completion scorer is applied
to each completion before averaging.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from lighteval.metrics.metrics_sample import (
    AvgAtN,
    GPassAtK,
    MajAtN,
    PassAtK,
    SampleLevelComputation,
    SamplingMetric,
)
from lighteval.metrics.utils.metric_utils import Metric, SampleLevelMetric, SamplingMethod
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


class GenerativeChoice(SampleLevelComputation):
    """Score one generated multiple-choice answer against ``Doc.gold_index``.

    This is used only for the avg branch of tasks whose native metric is
    log-likelihood based.  The native LOGPROBS metric remains unchanged.
    """

    _ANSWER_RE = re.compile(
        r"(?is)(?:answer|option|choice)(?:\s+is)?\s*[:：]?\s*[\(\[]?([A-Z])\b"
    )
    _PREFIX_RE = re.compile(r"(?is)^\s*[\(\[]?([A-Z])(?:[\)\].:,\s]|$)")

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> float:
        del kwargs
        predictions = list(getattr(model_response, "final_text", []) or [])
        if not predictions:
            return 0.0

        choices = [str(choice).strip() for choice in (doc.choices or [])]
        gold_indices = {int(index) for index in as_list(doc.gold_index)}
        if not choices or not gold_indices:
            return 0.0

        prediction = str(predictions[0]).strip()
        normalized_prediction = prediction.casefold()
        for index, choice in enumerate(choices):
            normalized_choice = choice.casefold()
            if normalized_prediction == normalized_choice:
                return float(index in gold_indices)

        match = self._ANSWER_RE.search(prediction) or self._PREFIX_RE.search(prediction)
        if match is not None:
            label = match.group(1).upper()
            index = ord(label) - ord("A")
            return float(index in gold_indices and index < len(choices))

        # A bare single-letter answer is common after the answer cue.
        if len(normalized_prediction) == 1 and "a" <= normalized_prediction <= "z":
            index = ord(normalized_prediction.upper()) - ord("A")
            return float(index in gold_indices and index < len(choices))
        return 0.0


class _SinglePredictionScorer(SampleLevelComputation):
    """Expose one scalar score from a sampling metric's internal scorer."""

    def __init__(self, source: SamplingMetric | Any):
        self.source = source

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs: Any) -> float:
        del kwargs
        predictions = list(getattr(model_response, "final_text", []) or [])
        if not predictions:
            return 0.0

        if hasattr(self.source, "score_one"):
            return float(self.source.score_one(doc, predictions[0]))

        source = self.source
        processed_doc = copy.deepcopy(doc)
        if isinstance(source, SamplingMetric):
            processed_doc.choices = [source.preprocess(text=str(choice)) for choice in doc.choices]
            prediction = source.preprocess(text=predictions[0])
        else:
            prediction = predictions[0]
        single_response = ModelResponse(text=[prediction])
        if isinstance(source, SampleLevelComputation):
            # Native sample scorers are not required to put ``doc`` first in
            # their Python signature (LCB's CodegenMetric is one example).
            # AvgAtN invokes the per-rollout scorer through this adapter, so
            # use the public keyword contract instead of positional arguments.
            return float(source.compute(doc=processed_doc, model_response=single_response))
        return float(source.compute_score(doc=processed_doc, model_response=single_response))


def build_avg_at_n_metric(metric: Metric, *, k: int, name: str = "avg@k") -> Metric:
    """Build a real generative avg metric from a task metric.

    ``k`` controls both ``AvgAtN.n`` and, through LightEval's task sampling
    logic, the number of completions requested from the model.
    """

    if int(k) <= 0:
        raise ValueError("avg@k requires a positive k")
    k = int(k)
    sample_fn = getattr(metric, "sample_level_fn", None)
    category = getattr(metric, "category", None)
    metric_names = getattr(metric, "metric_name", None)

    if isinstance(sample_fn, AvgAtN):
        cloned = copy.deepcopy(metric)
        cloned.sample_level_fn.n = k
        cloned.metric_name = name
        return cloned

    if isinstance(sample_fn, (PassAtK, MajAtN, GPassAtK)) or hasattr(sample_fn, "score_one"):
        scorer: Callable | SampleLevelComputation = _SinglePredictionScorer(sample_fn)
    elif category == SamplingMethod.LOGPROBS:
        scorer = GenerativeChoice()
    # Keep unrelated grouped generative metrics intact.  Targeted logprob
    # groups, such as TruthfulQA MC, use the generic choice scorer above.
    elif isinstance(metric_names, (list, tuple)):
        return copy.deepcopy(metric)
    elif isinstance(sample_fn, SampleLevelComputation):
        # Keep the native scorer's ``compute(doc=..., model_response=...)``
        # call contract. Calling a copied scorer positionally breaks native
        # scorers whose signature is ``compute(model_response, doc)``.
        scorer = _SinglePredictionScorer(copy.deepcopy(sample_fn))
    else:
        raise ValueError(
            f"cannot create real avg@{k} for metric {getattr(metric, 'metric_name', metric)!r}"
        )

    return SampleLevelMetric(
        metric_name=name,
        sample_level_fn=AvgAtN(n=k, sample_scoring_function=scorer),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
        batched_compute=False,
    )


__all__ = ["GenerativeChoice", "build_avg_at_n_metric"]
