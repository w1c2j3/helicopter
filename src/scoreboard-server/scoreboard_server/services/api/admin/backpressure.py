from __future__ import annotations

from scoreboard_server.dtos.api.admin.backpressure import AdminBackpressureResponse
from scoreboard_server.cores.scheduler_admin import controller


def admin_backpressure_response(*, infer_base_url: str | None) -> AdminBackpressureResponse:
    return controller.backpressure(infer_base_url)
