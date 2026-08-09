from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from scoreboard_server.routes.api.eval_records import register
from scoreboard_server.services.api.eval_records import eval_records_response


class _FakeStore:
    def __init__(self) -> None:
        self.records = [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "has_eval_record": True,
                "is_passed": True,
                "answer": "A",
                "ref_answer": "A",
                "fail_reason": "",
                "context_preview": "",
                "final_stop_reason": "stop_token",
                "final_stop_telemetry_observed": True,
                "is_truncated": False,
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "has_eval_record": True,
                "is_passed": False,
                "answer": "",
                "ref_answer": "B",
                "fail_reason": "missing_prediction",
                "context_preview": "",
                "final_stop_reason": "max_tokens",
                "final_stop_telemetry_observed": True,
                "is_truncated": True,
            },
            {
                "sample_index": 2,
                "repeat_index": 0,
                "pass_index": 0,
                "has_eval_record": False,
                "is_passed": None,
                "answer": None,
                "ref_answer": None,
                "fail_reason": None,
                "context_preview": "",
                "final_stop_reason": None,
                "final_stop_telemetry_observed": False,
                "is_truncated": False,
            },
        ]

    async def list_eval_records_for_space(self, **kwargs):  # noqa: ANN003, ANN202
        assert kwargs["limit"] is None
        assert kwargs["offset"] == 0
        assert kwargs["only_wrong"] is False
        return list(self.records)


async def test_all_mode_returns_every_completion_and_task_wide_diagnostics() -> None:
    response = await eval_records_response(
        _FakeStore(),  # type: ignore[arg-type]
        task_id=7,
        only_wrong=False,
        limit=None,
        offset=0,
    )

    assert len(response["records"]) == 3
    assert response["total"] == 3
    assert response["completion_total"] == 3
    assert response["eval_total"] == 2
    assert response["missing_eval_count"] == 1
    assert response["outcome"] == "all"
    assert response["outcome_counts"] == {
        "all": 3,
        "correct": 1,
        "incorrect": 0,
        "unanswered": 2,
    }
    assert response["has_more"] is False
    assert response["diagnostics"] == {
        "blank_count": 1,
        "blank_rate": 0.5,
        "missing_prediction_count": 1,
        "missing_prediction_rate": 0.5,
        "truncated_count": 1,
        "truncation_rate": 1 / 3,
        "final_stop_telemetry_count": 2,
        "conditional_truncation_rate": 0.5,
        "missing_eval_count": 1,
    }


async def test_explicit_limit_and_wrong_filter_remain_paginated() -> None:
    response = await eval_records_response(
        _FakeStore(),  # type: ignore[arg-type]
        task_id=7,
        only_wrong=True,
        limit=1,
        offset=0,
    )

    assert [record["sample_index"] for record in response["records"]] == [1]
    assert response["filtered_total"] == 1
    assert response["next_offset"] == 1
    assert response["has_more"] is False
    # Diagnostics describe the whole task, not only the filtered page.
    assert response["completion_total"] == 3
    assert response["diagnostics"]["truncation_rate"] == 1 / 3


async def test_outcome_filter_is_applied_before_pagination() -> None:
    store = _FakeStore()
    store.records.append(
        {
            "sample_index": 3,
            "repeat_index": 0,
            "pass_index": 0,
            "has_eval_record": True,
            "is_passed": False,
            "answer": "C",
            "ref_answer": "D",
            "fail_reason": "wrong_answer",
            "context_preview": "",
            "final_stop_reason": "stop_token",
            "final_stop_telemetry_observed": True,
            "is_truncated": False,
        }
    )

    first_unanswered = await eval_records_response(
        store,  # type: ignore[arg-type]
        task_id=7,
        only_wrong=False,
        outcome="unanswered",
        limit=1,
        offset=0,
    )
    second_unanswered = await eval_records_response(
        store,  # type: ignore[arg-type]
        task_id=7,
        only_wrong=False,
        outcome="unanswered",
        limit=1,
        offset=1,
    )
    incorrect = await eval_records_response(
        store,  # type: ignore[arg-type]
        task_id=7,
        only_wrong=False,
        outcome="incorrect",
        limit=20,
        offset=0,
    )

    assert [row["sample_index"] for row in first_unanswered["records"]] == [1]
    assert first_unanswered["filtered_total"] == 2
    assert first_unanswered["has_more"] is True
    assert [row["sample_index"] for row in second_unanswered["records"]] == [2]
    assert second_unanswered["has_more"] is False
    assert [row["sample_index"] for row in incorrect["records"]] == [3]
    assert incorrect["outcome_counts"] == {
        "all": 4,
        "correct": 1,
        "incorrect": 1,
        "unanswered": 2,
    }


async def test_route_omitting_limit_selects_all_mode() -> None:
    app = FastAPI()
    register(app, _FakeStore())  # type: ignore[arg-type]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = (await client.get("/api/eval-records", params={"task_id": 7})).json()

    assert payload["limit"] is None
    assert payload["total"] == 3
    assert len(payload["records"]) == 3

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        large_page = (
            await client.get("/api/eval-records", params={"task_id": 7, "limit": 10_000})
        )

    assert large_page.status_code == 200
    assert large_page.json()["limit"] == 10_000

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        unanswered = await client.get(
            "/api/eval-records",
            params={"task_id": 7, "outcome": "unanswered", "limit": 1},
        )
        invalid = await client.get(
            "/api/eval-records",
            params={"task_id": 7, "outcome": "not-a-category"},
        )

    assert unanswered.status_code == 200
    assert unanswered.json()["filtered_total"] == 2
    assert unanswered.json()["has_more"] is True
    assert invalid.status_code == 422
