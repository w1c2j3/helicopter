from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.cancel import AdminEvalCancelResponse


def admin_eval_cancel_response() -> AdminEvalCancelResponse:
    return controller.cancel()
