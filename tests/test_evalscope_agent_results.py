from __future__ import annotations

import json

from helicopter_cli.evalscope_agent_results import (
    discriminate_agent_result,
    extract_agent_answer,
    write_trace_report,
)


def test_choice_extraction_requires_an_explicit_final_choice() -> None:
    result = extract_agent_answer("I considered A and B.\nThe answer is C.", format_kind="choice")
    assert result.status == "ok"
    assert result.extracted_answer == " C"

    failed = extract_agent_answer("The reasoning mentions A, B, and C.", format_kind="choice")
    assert failed.status == "extraction_failed"
    assert failed.extracted_answer is None


def test_numeric_extraction_and_strict_comparison() -> None:
    extraction = extract_agent_answer("The calculation is complete.\nFinal answer: 1,000", format_kind="numeric")
    assert extraction.status == "ok"
    decision = discriminate_agent_result(extraction, reference_answer="$1000$")
    assert decision.status == "correct"


def test_short_answer_requires_marker_and_distinguishes_model_error() -> None:
    extraction = extract_agent_answer("Exact Answer: Ada Lovelace", format_kind="short_answer")
    assert discriminate_agent_result(extraction, reference_answer="Grace Hopper").status == "model_error"
    assert extract_agent_answer("Ada Lovelace", format_kind="short_answer").status == "extraction_failed"


def test_code_and_structured_answers_are_not_repaired() -> None:
    code = extract_agent_answer("```python\nprint(1)\n```", format_kind="code")
    assert code.status == "ok"
    assert code.extracted_answer == "print(1)"
    assert extract_agent_answer("print(1)", format_kind="code").status == "format_invalid"

    structured = extract_agent_answer('```json\n{"b": 2, "a": 1}\n```', format_kind="structured")
    assert structured.status == "ok"
    assert json.loads(structured.extracted_answer or "") == {"a": 1, "b": 2}
    assert extract_agent_answer('{"a": 1,', format_kind="structured").status == "format_invalid"


def test_swe_bench_backticks_and_tool_call_failures_are_explicit() -> None:
    action = extract_agent_answer("thought\n```mswea_bash_command\npytest -q\n```", format_kind="swe_bench_backticks")
    assert action.status == "ok"
    assert action.extracted_answer == "pytest -q"
    assert extract_agent_answer("pytest -q", format_kind="swe_bench_backticks").status == "format_invalid"

    missing_tool_call = extract_agent_answer("I should call bash.", format_kind="function_calling")
    assert discriminate_agent_result(missing_tool_call).status == "format_invalid"


def test_transport_and_context_failures_precede_answer_scoring() -> None:
    extraction = extract_agent_answer("Final answer: 2", format_kind="numeric")
    assert discriminate_agent_result(extraction, reference_answer="2", transport_status=400).status == "interface_error"
    assert discriminate_agent_result(extraction, reference_answer="2", finish_reason="length").status == "context_truncated"


def test_trace_report_keeps_transport_classification_and_raw_content(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "path": "/v1/chat/completions",
                "request": {"json": {"tools": [{"type": "function"}]}},
                "response": {
                    "status": 200,
                    "body": {"choices": [{"finish_reason": "length", "message": {"content": "raw"}}]},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    write_trace_report(trace, output, exit_code=0)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"] == {"context_truncated": 1}
    assert report["items"][0]["decision"]["raw_response"] == "raw"
