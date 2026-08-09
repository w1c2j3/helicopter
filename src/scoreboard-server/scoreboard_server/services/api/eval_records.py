from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.eval_records import EvalRecordsResponse


EvalRecordOutcome = Literal["all", "correct", "incorrect", "unanswered"]
_UNANSWERED_PATTERN = re.compile(r"empty|unanswer|no answer|truncat|max length|未作答|截断")


def _record_outcome(record: Mapping[str, object]) -> EvalRecordOutcome:
    if record.get("is_passed") is True:
        return "correct"
    answer = str(record.get("answer") or "")
    diagnostic = f"{answer} {record.get('fail_reason') or ''}".lower()
    if not answer.strip() or _UNANSWERED_PATTERN.search(diagnostic):
        return "unanswered"
    return "incorrect"


async def eval_records_response(
    store: ScoreboardStore,
    *,
    task_id: int,
    only_wrong: bool,
    limit: int | None,
    offset: int,
    outcome: EvalRecordOutcome = "all",
) -> EvalRecordsResponse:
    records = await store.list_eval_records_for_space(
        task_id=str(task_id),
        only_wrong=False,
        limit=None,
        offset=0,
        include_context=False,
        include_preview=limit is not None,
    )

    completion_total = len(records)
    eval_total = sum(bool(record.get("has_eval_record")) for record in records)
    missing_eval_count = completion_total - eval_total
    blank_count = sum(
        bool(record.get("has_eval_record"))
        and not str(record.get("answer") or "").strip()
        for record in records
    )
    missing_prediction_count = sum(
        bool(record.get("has_eval_record"))
        and str(record.get("fail_reason") or "").strip() == "missing_prediction"
        for record in records
    )
    truncated_count = sum(bool(record.get("is_truncated")) for record in records)
    final_stop_telemetry_count = sum(
        bool(record.get("final_stop_telemetry_observed")) for record in records
    )

    record_outcomes = [_record_outcome(record) for record in records]
    outcome_counts = {
        "all": completion_total,
        "correct": sum(value == "correct" for value in record_outcomes),
        "incorrect": sum(value == "incorrect" for value in record_outcomes),
        "unanswered": sum(value == "unanswered" for value in record_outcomes),
    }

    if only_wrong:
        # Backward compatibility: the legacy flag means every persisted eval
        # explicitly marked failed, including blank/unanswered predictions.
        filtered = [
            record
            for record in records
            if bool(record.get("has_eval_record")) and record.get("is_passed") is False
        ]
    elif outcome == "all":
        filtered = records
    else:
        filtered = [
            record
            for record, record_outcome in zip(records, record_outcomes, strict=True)
            if record_outcome == outcome
        ]
    filtered_total = len(filtered)
    if limit is None:
        page = filtered[offset:]
    else:
        page = filtered[offset : offset + limit]
    next_offset = offset + len(page)

    return {
        "task_id": task_id,
        "records": page,
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "has_more": next_offset < filtered_total,
        # ``total`` deliberately follows the records collection: unfiltered
        # evidence is completion-complete even when an eval row is missing.
        "total": completion_total,
        "filtered_total": filtered_total,
        "completion_total": completion_total,
        "eval_total": eval_total,
        "missing_eval_count": missing_eval_count,
        "outcome": outcome,
        "outcome_counts": outcome_counts,
        "diagnostics": {
            "blank_count": blank_count,
            "blank_rate": blank_count / eval_total if eval_total else None,
            "missing_prediction_count": missing_prediction_count,
            "missing_prediction_rate": (
                missing_prediction_count / eval_total if eval_total else None
            ),
            "truncated_count": truncated_count,
            # Primary strict-46 rate: final-stage truncations divided by every
            # persisted completion. Missing telemetry is not treated as clean.
            "truncation_rate": (
                truncated_count / completion_total if completion_total else None
            ),
            "final_stop_telemetry_count": final_stop_telemetry_count,
            "conditional_truncation_rate": (
                truncated_count / final_stop_telemetry_count
                if final_stop_telemetry_count
                else None
            ),
            "missing_eval_count": missing_eval_count,
        },
    }
