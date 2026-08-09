from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from scoreboard_server.db.connection import check_db_readiness
from scoreboard_server.routes.api import health as health_route
from scoreboard_server.services.api import health as health_service


@pytest.mark.asyncio
async def test_db_readiness_checks_connection_and_core_table(monkeypatch: pytest.MonkeyPatch) -> None:
    init_db = AsyncMock()
    execute_query = AsyncMock(return_value=(1, [{"?column?": 1}]))
    connection = Mock(execute_query=execute_query)
    values_list = AsyncMock(return_value=[])
    task_query = Mock()
    task_query.limit.return_value = task_query
    task_query.values_list = values_list

    monkeypatch.setattr("scoreboard_server.db.connection.init_db", init_db)
    monkeypatch.setattr(
        "scoreboard_server.db.connection.Tortoise.get_connection",
        Mock(return_value=connection),
    )
    monkeypatch.setattr("scoreboard_server.db.models.Task.all", Mock(return_value=task_query))

    settings = Mock()
    await check_db_readiness(settings)

    init_db.assert_awaited_once_with(settings)
    execute_query.assert_awaited_once_with("SELECT 1")
    task_query.limit.assert_called_once_with(1)
    values_list.assert_awaited_once_with("task_id", flat=True)


@pytest.mark.asyncio
async def test_health_keeps_success_response_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    readiness = AsyncMock()
    monkeypatch.setattr(health_service, "check_db_readiness", readiness)
    store = Mock(settings=Mock())

    assert await health_service.health_response(store) == {"status": "ok"}
    readiness.assert_awaited_once_with(store.settings)


@pytest.mark.asyncio
async def test_health_returns_sanitized_503_when_database_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(_store: object) -> dict[str, str]:
        raise RuntimeError("password=secret host=private-db")

    monkeypatch.setattr(health_route, "health_response", unavailable)
    app = FastAPI()
    health_route.register(app, Mock())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "service unavailable"}
    assert "secret" not in response.text
    assert "private-db" not in response.text
