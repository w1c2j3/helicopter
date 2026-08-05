from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.routes.api import register_api_routes
from scoreboard_server.services.api.admin.auth import install_admin_auth


def create_app(
    settings: DatabaseSettings | None = None,
    *,
    generate_schemas: bool | None = None,
) -> FastAPI:
    resolved = settings or DatabaseSettings.from_env()
    should_generate_schemas = generate_schemas
    if should_generate_schemas is None:
        should_generate_schemas = os.environ.get("SCOREBOARD_GENERATE_SCHEMAS", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db(resolved, generate_schemas=should_generate_schemas)
        try:
            yield
        finally:
            await close_db()

    app = FastAPI(title="Helicopter Scoreboard", version="0.1.0", lifespan=lifespan)
    install_admin_auth(app)
    cors_origins = [
        item.strip().rstrip("/")
        for item in (os.environ.get("SCOREBOARD_CORS_ORIGINS") or "").split(",")
        if item.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Accept", "Content-Type", "X-RWKV-Admin-Request"],
        )
    store = ScoreboardStore(settings=resolved)
    register_api_routes(app, store)
    return app


app = create_app()
