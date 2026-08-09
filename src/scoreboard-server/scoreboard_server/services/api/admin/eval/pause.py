from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.pause import AdminEvalPauseResponse


def admin_eval_pause_response() -> AdminEvalPauseResponse:
    return controller.pause()
