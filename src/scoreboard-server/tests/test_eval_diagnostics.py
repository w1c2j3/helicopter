from __future__ import annotations

from scoreboard_server.cores.eval_diagnostics import final_generation_diagnostics


def test_math_ignores_truncated_reasoning_stage() -> None:
    result = final_generation_diagnostics(
        {
            "stages": [
                {"completion": "reasoning", "stop_reason": "max_tokens"},
                {"completion": "42", "stop_reason": "stop_token"},
            ],
            "stats": {
                "stage1": {"truncated": True},
                "stage2": {"truncated": False},
            },
        },
        domain="math",
    )

    assert result == {
        "final_stop_reason": "stop_token",
        "final_stop_telemetry_observed": True,
        "is_truncated": False,
    }


def test_math_never_falls_back_to_stage_one() -> None:
    result = final_generation_diagnostics(
        {"stages": [{"completion": "reasoning", "stop_reason": "max_tokens"}]},
        domain="math",
    )

    assert result["final_stop_telemetry_observed"] is False
    assert result["is_truncated"] is False


def test_math_uses_final_stage_stats_or_reason() -> None:
    result = final_generation_diagnostics(
        {
            "stages": [
                {"completion": "reasoning", "stop_reason": "stop_token"},
                {"completion": "42", "stop_reason": "stop_token"},
            ],
            "stats": {"stage2": {"truncated": True}},
        },
        domain="math",
    )

    assert result["final_stop_reason"] == "stop_token"
    assert result["is_truncated"] is True


def test_knowledge_uses_evaluator_facing_answer_bridge() -> None:
    result = final_generation_diagnostics(
        {
            "stages": [{"completion": "reasoning", "stop_reason": "max_tokens"}],
            "format_bridges": {
                "answer_stage_raw_completion": " B",
                "answer_stage_raw_stop_reason": "length",
            },
        },
        domain="knowledge",
    )

    # A complete one-letter answer at the token boundary is not a semantically
    # truncated choice, matching the strict-46 matrix query.
    assert result["final_stop_reason"] == "length"
    assert result["final_stop_telemetry_observed"] is True
    assert result["is_truncated"] is False


def test_knowledge_marks_incomplete_raw_answer_as_truncated() -> None:
    result = final_generation_diagnostics(
        {
            "format_bridges": {
                "strategy_b_final_raw_completion": "The answer appears to be",
                "strategy_b_final_raw_stop_reason": "max_length",
            }
        },
        domain="knowledge",
    )

    assert result["is_truncated"] is True


def test_ordinary_generation_uses_direct_reason_precedence() -> None:
    result = final_generation_diagnostics(
        {
            "direct_raw_finish_reason": "stop_token",
            "stages": [{"completion": "code", "stop_reason": "max_tokens"}],
        },
        domain="coding",
    )

    assert result["final_stop_reason"] == "stop_token"
    assert result["is_truncated"] is False
