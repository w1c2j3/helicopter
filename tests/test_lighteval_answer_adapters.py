from __future__ import annotations

import tomllib
from pathlib import Path

from lighteval.metrics.avg_at_n import GenerativeChoice, build_avg_at_n_metric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import AvgAtN
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc

from helicopter_cli.lighteval_answer_adapters import (
    adapt_answer,
    answers_match,
    extract_choice_answer,
    extract_code_completion,
    extract_math_answer,
    normalize_math_answer,
)
from helicopter_cli.lighteval_g1h_policy import _normalize_doc_references


def test_choice_adapter_matches_rwkv_direct_letter_and_cot_answer() -> None:
    prompt = "Question\nA. first\nB. second\nC. third\nD. fourth"
    assert extract_choice_answer(" B", prompt=prompt) == " B"
    assert extract_choice_answer("reasoning mentions A\nFinal answer: D.", prompt=prompt) == " D"
    assert extract_choice_answer("<think>reasoning</think>\nC", prompt=prompt) == " C"


def test_choice_adapter_accepts_common_explicit_choice_cues() -> None:
    assert extract_choice_answer("Answer Choice: (D) 2/5") == " D"
    assert extract_choice_answer("The correct choice is (D) 11.5.") == " D"
    assert extract_choice_answer("**Correct choice: (B) After 12 minutes**") == " B"
    assert extract_choice_answer("Thus, the capacity corresponds to option (D).") == " D"


def test_choice_adapter_does_not_read_answer_choices_header_as_answer() -> None:
    value = "Answer Choices: (A) 7.5 (B) 8.9 (C) 9.9 (D) 11.5 (E) 11.7"
    assert extract_choice_answer(value) != " A"
    assert extract_choice_answer(" E") == " E"


def test_choice_adapter_does_not_use_an_arbitrary_reasoning_tail() -> None:
    assert extract_choice_answer("reasoning without a final option") == ""


def test_choice_adapter_does_not_treat_roman_numerals_as_option_labels() -> None:
    prompt = (
        "Question\n"
        "I. first statement\n"
        "II. second statement\n"
        "Answer Choices: (A)I only (B)II only (C)both"
    )
    assert extract_choice_answer(" A", prompt=prompt) == " A"


def test_math_adapter_prefers_the_last_boxed_answer() -> None:
    assert extract_math_answer("work\n\\boxed{3}\nmore work\n\\boxed{42}") == "$42$"


def test_math_adapter_falls_back_to_boxed_value_before_closing_think() -> None:
    assert extract_math_answer("reasoning\n\\boxed{42}</think>") == "$42$"
    assert extract_math_answer("\\boxed{3}\nreasoning\n\\boxed{42}</think>") == "$42$"
    assert extract_math_answer("\\boxed{3}\nreasoning\n\\boxed{42") == "$3$"
    assert extract_math_answer("reasoning\n\\boxed{42") == ""


def test_math_adapter_extracts_final_delimited_answer_after_think() -> None:
    completion = (
        "long reasoning with the answer 540 before the boundary"
        "</think>"
        r"The maximum is \(\sqrt{324^2 + 432^2}=540\)."
    )
    assert extract_math_answer(completion) == "$540$"
    assert answers_match(
        completion,
        r"$540$",
        domain="math",
        request_format="math_boxed",
    ) is True


def test_code_adapter_keeps_the_last_program_block() -> None:
    text = "explanation\n```text\nnot code\n```\n```python\ndef f():\n    return 1\n```"
    assert extract_code_completion(text) == "```\ndef f():\n    return 1\n```"


def test_choice_adapter_is_symmetric_for_model_and_reference_text() -> None:
    prompt = "Question\nA. first\nB. second"
    assert adapt_answer("Final answer: B", domain="knowledge", request_format="choice", prompt=prompt) == " B"
    assert adapt_answer("B", domain="knowledge", request_format="choice", prompt=prompt) == " B"


def test_choice_format_overrides_broad_math_domain() -> None:
    # AGIEval aqua-rat is classified as math but scored as a choice task.
    assert answers_match(" C", "C", domain="math", request_format="choice") is True


def test_math_adapter_is_symmetric_for_latex_wrappers() -> None:
    values = [r"$\boxed{4000}$", r"\boxed{4000}", r"$4000$", "4000"]
    assert {adapt_answer(value, domain="math", request_format="math_boxed") for value in values} == {"$4000$"}
    assert adapt_answer(r"$1/2$", domain="math", request_format="math_boxed") == r"$\frac{1}{2}$"
    assert adapt_answer(r"\frac{1}{2}", domain="math", request_format="math_boxed") == r"$\frac{1}{2}$"


def test_math_adapter_falls_back_from_truncated_fraction_to_previous_answer() -> None:
    assert adapt_answer("Final answer: 42\nFinal answer: \\frac", domain="math", request_format="math_boxed") == "$42$"
    assert adapt_answer(r"Final answer: \frac", domain="math", request_format="math_boxed") == ""
    assert adapt_answer(r"Final answer: \frac{1}", domain="math", request_format="math_boxed") == ""
    assert adapt_answer("\\boxed{42}\n\\boxed{\\frac}", domain="math", request_format="math_boxed") == "$42$"


def test_math_adapter_rejects_prose_and_uses_the_last_valid_cue() -> None:
    assert adapt_answer(
        "The answer is 25.\nsubstituting into the second equation:",
        domain="math",
        request_format="math_boxed",
    ) == "$25$"
    assert adapt_answer(
        "substituting into the second equation:",
        domain="math",
        request_format="math_boxed",
    ) == ""
    assert adapt_answer(
        r"reasoning\nThe product \(\prod_{k=0}^{12}(2-2\omega^k)\) is evaluated repeatedly",
        domain="math",
        request_format="math_boxed",
    ) == ""


def test_math_adapter_normalizes_integer_leading_zeros_symmetrically() -> None:
    assert normalize_math_answer(r"$025$") == "$25$"
    assert answers_match(
        r"$025$",
        r"\boxed{25}",
        domain="math",
        request_format="math_boxed",
    ) is True


def test_code_adapter_is_symmetric_for_fenced_and_plain_reference() -> None:
    expected = "```\ndef f():\n    return 1\n```"
    assert adapt_answer("def f():\n    return 1", domain="coding", request_format="python_program") == expected
    assert adapt_answer("```python\ndef f():\n    return 1\n```", domain="coding", request_format="python_program") == expected


def test_code_answers_defer_to_native_execution_scorer() -> None:
    assert answers_match(
        "def f():\n    return 1",
        "",
        domain="coding",
        request_format="python_program",
    ) is None


def test_official_avg_at_n_scores_adapted_math_choice_and_code_rollouts() -> None:
    math_doc = Doc(query="q", choices=[r"\boxed{4000}"], gold_index=0)
    _normalize_doc_references(math_doc, domain="math", request_format="math_boxed")
    math_response = ModelResponse(
        text=[r"$4000$", r"\boxed{4000}"],
        text_post_processed=[
            adapt_answer(value, domain="math", request_format="math_boxed")
            for value in (r"$4000$", r"\boxed{4000}")
        ],
    )
    math_avg = AvgAtN(
        n=2,
        sample_scoring_function=Metrics.exact_match.value.sample_level_fn,
    )
    assert math_avg.compute(math_doc, math_response) == 1.0

    choice_doc = Doc(query="A. first\nB. second", choices=["first", "second"], gold_index=1)
    choice_response = ModelResponse(
        text=["B", "Final answer: B"],
        text_post_processed=[
            adapt_answer(value, domain="knowledge", request_format="choice", prompt=choice_doc.query)
            for value in ("B", "Final answer: B")
        ],
    )
    choice_avg = AvgAtN(n=2, sample_scoring_function=GenerativeChoice())
    assert choice_avg.compute(choice_doc, choice_response) == 1.0

    code_doc = Doc(query="q", choices=["def f():\n    return 1"], gold_index=0)
    _normalize_doc_references(code_doc, domain="coding", request_format="python_program")
    code_response = ModelResponse(
        text=["def f():\n    return 1", "```python\ndef f():\n    return 1\n```"],
        text_post_processed=[
            adapt_answer(value, domain="coding", request_format="python_program")
            for value in ("def f():\n    return 1", "```python\ndef f():\n    return 1\n```")
        ],
    )
    code_avg = AvgAtN(
        n=2,
        sample_scoring_function=Metrics.exact_match.value.sample_level_fn,
    )
    assert code_avg.compute(code_doc, code_response) == 1.0


def test_avg_at_n_calls_native_sample_scorer_with_keyword_arguments() -> None:
    class NativeScorer(SampleLevelComputation):
        def compute(self, model_response, doc, **kwargs):
            del kwargs
            return float(doc.query == model_response.final_text[0])

    metric = SampleLevelMetric(
        metric_name="native",
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=NativeScorer(),
        corpus_level_fn=lambda values: sum(values) / len(values),
        higher_is_better=True,
        batched_compute=False,
    )
    wrapped = build_avg_at_n_metric(metric, k=2, name="avg@2")
    doc = Doc(query="good", choices=[""], gold_index=0)
    response = ModelResponse(text=["good", "bad"])
    assert wrapped.sample_level_fn.compute(doc, response) == 0.5


def test_instruction_adapter_keeps_the_complete_nocot_response() -> None:
    assert adapt_answer(
        "Answer with exactly two lines.\nfirst\nsecond",
        domain="instruction_following",
        request_format="instruction",
    ) == "Answer with exactly two lines.\nfirst\nsecond"


def test_large_suite_only_adds_cot_to_math_and_knowledge() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "configs/large_eval_60.toml").read_text(encoding="utf-8"))
    run = manifest["run"]
    assert run["base_modes"] == ["naive_nocot", "normal_nocot"]
    assert run["cot_fields"] == ["math", "knowledge"]
    assert run["cot_modes"] == ["naive_cot", "normal_cot"]
