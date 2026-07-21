from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from scoreboard_server.dtos.api.admin.eval.resume import AdminEvalResumeResponse
from scoreboard_server.cores.scheduler_admin import SchedulerAdminError
from scoreboard_server.services.api.admin import check_admin_auth
from scoreboard_server.services.api.admin.eval.resume import admin_eval_resume_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/resume")
    async def admin_resume(authorization: str | None = Header(default=None)) -> AdminEvalResumeResponse:
        check_admin_auth(authorization)
        try:
            return admin_eval_resume_response()
        except SchedulerAdminError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
