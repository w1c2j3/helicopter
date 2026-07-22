from __future__ import annotations

from datetime import datetime

from scoreboard_server.cores.leaderboard import build_leaderboard_payload


def _entry(
    *,
    score_id: int,
    model: str,
    dataset: str,
    score: float,
    profile: str,
    field: str = "knowledge",
    is_param_search: bool = False,
    temperature: float | None = None,
) -> dict[str, object]:
    sampling: dict[str, object] = {"prompt_profile": profile}
    if temperature is not None:
        sampling["sampling_config"] = {"answer": {"temperature": temperature}}
    return {
        "score_id": score_id,
        "task_id": score_id,
        "cot": False,
        "cot_mode": "NoCoT",
        "metrics": {"accuracy": score},
        "created_at": datetime(2026, 7, 22, 8, score_id),
        "is_param_search": is_param_search,
        "model": model,
        "dataset": dataset,
        "samples": 100,
        "task": "lighteval",
        "sampling_config": sampling,
        "field": field,
    }


def test_primary_matrix_is_naive_only_with_models_as_rows() -> None:
    entries = [
        _entry(score_id=1, model="rwkv7-g1g-1.5b-weight-a", dataset="mmlu_test", score=0.6, profile="naive"),
        _entry(score_id=2, model="rwkv7-g1g-2.9b-weight-b", dataset="mmlu_test", score=0.8, profile="naive"),
        _entry(score_id=3, model="rwkv7-g1g-2.9b-weight-b", dataset="arc_test", score=0.7, profile="naive"),
        _entry(score_id=4, model="rwkv7-g1g-7.2b-normal", dataset="mmlu_test", score=0.99, profile="normal"),
    ]

    payload = build_leaderboard_payload(
        entries,
        selected_model=None,
        view="benchmark_detail_latest",
        tuning_entries=entries,
    )
    knowledge = payload["matrix"]["domains"][0]

    assert [item["key"] for item in payload["matrix"]["domains"]] == [
        "knowledge",
        "math",
        "coding",
        "agent",
        "instruction_following",
        "function_call",
    ]
    assert {column["label"] for column in knowledge["columns"]} == {"arc_test", "mmlu_test"}
    assert [row["model"] for row in knowledge["rows"]] == [
        "rwkv7-g1g-2.9b-weight-b",
        "rwkv7-g1g-1.5b-weight-a",
    ]
    assert all("normal" not in row["model"] for row in knowledge["rows"])


def test_normal_and_parameter_search_are_kept_on_tuning_matrix() -> None:
    entries = [
        _entry(score_id=1, model="rwkv7-g1g-1.5b-weight-a", dataset="mmlu_test", score=0.6, profile="naive"),
        _entry(score_id=2, model="rwkv7-g1g-1.5b-weight-a", dataset="mmlu_test", score=0.7, profile="normal", temperature=0.2),
        _entry(
            score_id=3,
            model="rwkv7-g1g-2.9b-weight-b",
            dataset="mmlu_test",
            score=0.8,
            profile="normal",
            is_param_search=True,
            temperature=0.8,
        ),
    ]

    payload = build_leaderboard_payload(
        entries,
        selected_model=None,
        view="benchmark_detail_latest",
        tuning_entries=entries,
    )
    tuning = payload["tuning_matrix"]["benchmarks"][0]

    assert payload["tuning_matrix"]["benchmark_count"] == 1
    assert len(tuning["columns"]) == 2
    assert {row["model"] for row in tuning["rows"]} == {
        "rwkv7-g1g-1.5b-weight-a",
        "rwkv7-g1g-2.9b-weight-b",
    }
