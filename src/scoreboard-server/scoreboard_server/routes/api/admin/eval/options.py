from __future__ import annotations

from fastapi import FastAPI, Header

from scoreboard_server.dtos.api.admin.eval.options import AdminEvalOptionsResponse
from scoreboard_server.services.api.admin.eval.options import admin_eval_options_response
from scoreboard_server.services.api.admin import check_admin_auth


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/options")
    async def admin_options(authorization: str | None = Header(default=None)) -> AdminEvalOptionsResponse:
        check_admin_auth(authorization)
        return admin_eval_options_response()
