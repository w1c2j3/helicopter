from __future__ import annotations

from scoreboard_server.dtos.api.admin.health import AdminHealthResponse
from scoreboard_server.cores.scheduler_admin import controller
from scoreboard_server.services.api.admin import auth_required


def admin_health_response() -> AdminHealthResponse:
    snapshot = controller.snapshot()
    return {
        "status": "ok",
        "active": snapshot["status"] not in {"idle", "completed", "cancelled", "failed"},
        "auth_required": auth_required(),
    }
