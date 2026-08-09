from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.draft import AdminEvalDraftResponse


def admin_eval_draft_response() -> AdminEvalDraftResponse:
    return controller.draft()
