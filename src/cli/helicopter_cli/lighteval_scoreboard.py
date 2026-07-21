from __future__ import annotations

import os

from lighteval.logging.evaluation_tracker import EvaluationTracker

from .scoreboard_bridge import write_lighteval_tracker


_ORIGINAL_SAVE = EvaluationTracker.save


def _database_only_save(self: EvaluationTracker) -> None:
    if os.environ.get("HELICOPTER_SCOREBOARD_DB_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        recorded = write_lighteval_tracker(self)
        for item in recorded:
            print(f"lighteval: database rows recorded: {item}")
        return
    _ORIGINAL_SAVE(self)


if not getattr(EvaluationTracker.save, "_helicopter_scoreboard_patch", False):
    _database_only_save._helicopter_scoreboard_patch = True  # type: ignore[attr-defined]
    EvaluationTracker.save = _database_only_save
