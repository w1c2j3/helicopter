from __future__ import annotations

import json
from datetime import datetime, timezone

from helicopter_cli.__main__ import build_parser
from helicopter_cli.evalscope_scoreboard import build_import_plan, cleanup_json_artifacts
from scoreboard_server.db.repository import ScoreboardStore


def _prediction(index: int, *, finish_reason: str = "stop") -> dict:
    return {
        "index": index,
        "model": "demo-model",
        "model_output": {
            "choices": [
                {
                    "message": {"role": "assistant", "content": f"answer-{index}"},
                    "finish_reason": finish_reason,
                }
            ],
            "perf_metrics": {"input_tokens": 512, "output_tokens": 24},
        },
        "messages": [{"role": "user", "content": f"question-{index}"}],
        "metadata": {"function": [{"name": "tool"}]},
    }


def _review(index: int, acc: int) -> dict:
    return {
        "index": index,
        "target": "official-target",
        "sample_score": {
            "score": {
                "value": {"acc": acc},
                "extracted_prediction": f"official-answer-{index}",
                "metadata": {"error_message": "wrong" if not acc else "None"},
            },
            "sample_metadata": {"ground_truth": [{"tool": {"x": [index]}}]},
        },
    }


def test_build_import_plan_keeps_official_raw_records_and_scores(tmp_path) -> None:
    model = "demo-model"
    for kind in ("predictions", "reviews"):
        (tmp_path / kind / model).mkdir(parents=True)
    (tmp_path / "reports" / model).mkdir(parents=True)
    (tmp_path / "reports" / model / "bfcl_v4.json").write_text(
        json.dumps({"dataset_name": "bfcl_v4", "model_name": model, "score": 0.5, "metrics": []}),
        encoding="utf-8",
    )
    predictions = [_prediction(0), _prediction(1, finish_reason="length")]
    reviews = [_review(0, 1), _review(1, 0)]
    for kind, rows in (("predictions", predictions), ("reviews", reviews)):
        (tmp_path / kind / model / "bfcl_v4_simple_python.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    plan = build_import_plan(tmp_path, model_name=model)

    assert plan.context_audit["samples"] == 2
    assert plan.context_audit["context_error_samples"] == 0
    assert len(plan.completion_payloads) == 2
    assert len(plan.eval_payloads) == 2
    assert plan.eval_payloads[0]["is_passed"] is True
    assert plan.eval_payloads[1]["is_passed"] is False
    assert plan.completion_payloads[0]["agent_result"]["prediction"]["messages"][0]["content"] == "question-0"
    assert plan.completion_payloads[1]["agent_result"]["official_reference"] == [{"tool": {"x": [1]}}]


def test_build_import_plan_accepts_official_passed_and_preserves_unscored_values(tmp_path) -> None:
    model = "verifier-model"
    for kind in ("predictions", "reviews"):
        (tmp_path / kind / model).mkdir(parents=True)
    (tmp_path / "reports" / model).mkdir(parents=True)
    (tmp_path / "reports" / model / "k2_verifier.json").write_text(
        json.dumps({"dataset_name": "k2_verifier", "model_name": model, "score": 0.25}),
        encoding="utf-8",
    )
    predictions = [_prediction(0), _prediction(1)]
    reviews = [
        {
            "sample_score": {
                "score": {
                    "value": {"passed": True},
                    "extracted_prediction": "official-answer-0",
                }
            }
        },
        {
            "sample_score": {
                "score": {
                    "value": {"trigger_similarity": 0.0, "schema_accuracy": 1.0},
                    "extracted_prediction": "official-answer-1",
                }
            }
        },
    ]
    for kind, rows in (("predictions", predictions), ("reviews", reviews)):
        (tmp_path / kind / model / "k2_verifier_default.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    plan = build_import_plan(tmp_path, model_name=model, benchmark="k2_verifier")

    assert plan.invalid_reviews == 0
    assert plan.unscored_reviews == 1
    assert len(plan.eval_payloads) == 1
    assert plan.completion_payloads[1]["status"] == "Completed"
    assert plan.completion_payloads[1]["agent_result"]["official_sample_value"] == {
        "trigger_similarity": 0.0,
        "schema_accuracy": 1.0,
    }


def test_output_token_cap_is_not_a_context_error(tmp_path) -> None:
    model = "cap-model"
    for kind in ("predictions", "reviews"):
        (tmp_path / kind / model).mkdir(parents=True)
    (tmp_path / "reports" / model).mkdir(parents=True)
    (tmp_path / "reports" / model / "general_fc.json").write_text(
        json.dumps({"dataset_name": "general_fc", "model_name": model, "score": 0.0}),
        encoding="utf-8",
    )
    prediction = _prediction(0, finish_reason="length")
    prediction["model_output"]["choices"][0]["message"]["content"] = "truncated"
    review = {"sample_score": {"score": {"value": {"passed": False}}}}
    for kind, row in (("predictions", prediction), ("reviews", review)):
        (tmp_path / kind / model / "general_fc_default.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    plan = build_import_plan(tmp_path, model_name=model, benchmark="general_fc")

    assert plan.context_audit["context_error_samples"] == 0


def test_provider_connection_error_is_not_scored_as_model_failure(tmp_path) -> None:
    model = "transport-error-model"
    for kind in ("predictions", "reviews"):
        (tmp_path / kind / model).mkdir(parents=True)
    (tmp_path / "reports" / model).mkdir(parents=True)
    (tmp_path / "reports" / model / "general_fc.json").write_text(
        json.dumps({"dataset_name": "general_fc", "model_name": model, "score": 0.0}),
        encoding="utf-8",
    )
    prediction = _prediction(0)
    prediction["model_output"]["error"] = "Connection error."
    review = {"sample_score": {"score": {"value": {"passed": False}}}}
    for kind, row in (("predictions", prediction), ("reviews", review)):
        (tmp_path / kind / model / "general_fc_default.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )

    plan = build_import_plan(tmp_path, model_name=model, benchmark="general_fc")

    assert plan.inference_error_samples == 1
    assert plan.context_audit["status_counts"] == {"inference_error": 1}
    assert plan.context_audit["context_error_samples"] == 0
    assert plan.completion_payloads[0]["status"] == "Failed"
    assert plan.eval_payloads == []


def test_cleanup_json_artifacts_is_scoped_to_evalscope_work_dir(tmp_path) -> None:
    keep = tmp_path / "logs" / "run.log"
    keep.parent.mkdir()
    keep.write_text("log", encoding="utf-8")
    (tmp_path / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "report.json").write_text("{}", encoding="utf-8")

    assert cleanup_json_artifacts(tmp_path) == 2
    assert keep.is_file()
    assert not (tmp_path / "predictions.jsonl").exists()


def test_evalscope_parser_exposes_db_only_import_flags() -> None:
    args = build_parser().parse_args(
        ["eval", "evalscope", "demo", "bfcl_v4", "--scoreboard", "--scoreboard-db-only"]
    )

    assert args.scoreboard is True
    assert args.scoreboard_db_only is True


def test_scoreboard_store_adapts_created_at_to_database_timestamp_type() -> None:
    value = datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc)

    legacy_store = ScoreboardStore()
    legacy_store._legacy_naive_timestamps = True
    assert legacy_store._db_created_at(value).tzinfo is None

    current_store = ScoreboardStore()
    current_store._legacy_naive_timestamps = False
    assert current_store._db_created_at(value).tzinfo == timezone.utc
