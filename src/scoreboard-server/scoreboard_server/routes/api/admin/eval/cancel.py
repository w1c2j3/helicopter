from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException

from scoreboard_server.dtos.api.admin.eval.cancel import AdminEvalCancelResponse
from scoreboard_server.cores.scheduler_admin import SchedulerAdminError
from scoreboard_server.services.api.admin import check_admin_auth
from scoreboard_server.services.api.admin.eval.cancel import admin_eval_cancel_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/cancel")
    async def admin_cancel(authorization: str | None = Header(default=None)) -> AdminEvalCancelResponse:
        check_admin_auth(authorization)
        try:
            return admin_eval_cancel_response()
        except SchedulerAdminError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
