from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.options import AdminEvalOptionsResponse


def admin_eval_options_response() -> AdminEvalOptionsResponse:
    return controller.options()
