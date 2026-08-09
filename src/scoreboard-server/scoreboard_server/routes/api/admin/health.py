from __future__ import annotations

from fastapi import FastAPI, Header

from scoreboard_server.dtos.api.admin.health import AdminHealthResponse
from scoreboard_server.services.api.admin.health import admin_health_response
from scoreboard_server.services.api.admin import check_admin_auth


def register(app: FastAPI) -> None:
    @app.get("/api/admin/health")
    async def admin_health(authorization: str | None = Header(default=None)) -> AdminHealthResponse:
        check_admin_auth(authorization)
        return admin_health_response()
