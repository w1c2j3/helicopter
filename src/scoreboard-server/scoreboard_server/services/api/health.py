from __future__ import annotations

import asyncio

from scoreboard_server.db.connection import check_db_readiness
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.health import HealthResponse

READINESS_TIMEOUT_SECONDS = 2.0


async def health_response(store: ScoreboardStore) -> HealthResponse:
    await asyncio.wait_for(
        check_db_readiness(store.settings),
        timeout=READINESS_TIMEOUT_SECONDS,
    )
    return {"status": "ok"}
