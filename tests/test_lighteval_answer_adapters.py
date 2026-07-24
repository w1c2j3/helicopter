from __future__ import annotations

import tomllib
from pathlib import Path

from helicopter_cli.lighteval_answer_adapters import (
    adapt_answer,
    extract_choice_answer,
    extract_code_completion,
    extract_math_answer,
)


def test_choice_adapter_matches_rwkv_direct_letter_and_cot_answer() -> None:
    prompt = "Question\nA. first\nB. second\nC. third\nD. fourth"
    assert extract_choice_answer(" B", prompt=prompt) == " B"
    assert extract_choice_answer("reasoning mentions A\nFinal answer: D.", prompt=prompt) == " D"
    assert extract_choice_answer("<think>reasoning</think>\nC", prompt=prompt) == " C"


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
    assert extract_math_answer("reasoning\n\\boxed{42") == ""


def test_code_adapter_keeps_the_last_program_block() -> None:
    text = "explanation\n```text\nnot code\n```\n```python\ndef f():\n    return 1\n```"
    assert extract_code_completion(text) == "```python\ndef f():\n    return 1\n```"


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
