from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException

from scoreboard_server.cores.scheduler_admin import SchedulerAdminError
from scoreboard_server.services.api.admin import check_admin_auth
from scoreboard_server.services.api.admin.eval.start import admin_eval_start_response


def register(app: FastAPI) -> None:
    @app.post("/api/admin/eval/start")
    async def admin_start(
        payload: dict[str, Any] | None = Body(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        check_admin_auth(authorization)
        try:
            return admin_eval_start_response(payload or {})
        except SchedulerAdminError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
