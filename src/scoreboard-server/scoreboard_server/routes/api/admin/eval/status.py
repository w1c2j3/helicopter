from __future__ import annotations

from fastapi import FastAPI, Header

from scoreboard_server.dtos.api.admin.eval.status import AdminEvalStatusResponse
from scoreboard_server.services.api.admin.eval.status import admin_eval_status_response
from scoreboard_server.services.api.admin import check_admin_auth


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/status")
    async def admin_status(authorization: str | None = Header(default=None)) -> AdminEvalStatusResponse:
        check_admin_auth(authorization)
        return admin_eval_status_response()
