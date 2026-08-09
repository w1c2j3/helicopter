from __future__ import annotations

from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.dtos.api.admin.eval.resume import AdminEvalResumeResponse


def admin_eval_resume_response() -> AdminEvalResumeResponse:
    return controller.resume()
