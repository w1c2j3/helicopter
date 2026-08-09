from __future__ import annotations

from fastapi import FastAPI, HTTPException

from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.dtos.api.health import HealthResponse
from scoreboard_server.services.api.health import health_response


def register(app: FastAPI, store: ScoreboardStore) -> None:
    @app.get("/api/health")
    async def health() -> HealthResponse:
        try:
            return await health_response(store)
        except Exception:  # noqa: BLE001 - sanitize every database/driver failure at the HTTP boundary.
            raise HTTPException(status_code=503, detail="service unavailable") from None
