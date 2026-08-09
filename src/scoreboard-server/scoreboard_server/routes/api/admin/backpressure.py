from __future__ import annotations

from fastapi import FastAPI, Header, Query

from scoreboard_server.dtos.api.admin.backpressure import AdminBackpressureResponse
from scoreboard_server.services.api.admin.backpressure import admin_backpressure_response
from scoreboard_server.services.api.admin import check_admin_auth


def register(app: FastAPI) -> None:
    @app.get("/api/admin/backpressure")
    async def admin_backpressure(
        infer_base_url: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> AdminBackpressureResponse:
        check_admin_auth(authorization)
        return admin_backpressure_response(infer_base_url=infer_base_url)
