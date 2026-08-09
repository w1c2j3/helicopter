from __future__ import annotations

from typing import Any

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.start import AdminEvalStartResponse


def admin_eval_start_response(payload: dict[str, Any]) -> AdminEvalStartResponse:
    return controller.start(payload)
