from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.status import AdminEvalStatusResponse


def admin_eval_status_response() -> AdminEvalStatusResponse:
    return controller.snapshot()
