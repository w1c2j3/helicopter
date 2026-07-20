from __future__ import annotations

from fastapi import FastAPI, Header

from scoreboard_server.dtos.api.admin.eval.draft import AdminEvalDraftResponse
from scoreboard_server.services.api.admin.eval.draft import admin_eval_draft_response
from scoreboard_server.services.api.admin import check_admin_auth


def register(app: FastAPI) -> None:
    @app.get("/api/admin/eval/draft")
    async def admin_draft(authorization: str | None = Header(default=None)) -> AdminEvalDraftResponse:
        check_admin_auth(authorization)
        return admin_eval_draft_response()
