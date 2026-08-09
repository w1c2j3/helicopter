from __future__ import annotations

from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from scoreboard_server.routes.api.admin import auth as auth_routes
from scoreboard_server.services.api.admin import check_admin_auth
from scoreboard_server.services.api.admin.auth import install_admin_auth, password_hash


def _app() -> FastAPI:
    app = FastAPI()
    install_admin_auth(app)
    auth_routes.register(app)

    @app.get("/api/admin/protected")
    async def protected_get(authorization: str | None = Header(default=None)) -> dict[str, bool]:
        check_admin_auth(authorization)
        return {"ok": True}

    @app.post("/api/admin/protected")
    async def protected_post(authorization: str | None = Header(default=None)) -> dict[str, bool]:
        check_admin_auth(authorization)
        return {"ok": True}

    return app


def test_admin_password_session_is_backend_only(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_ADMIN_AUTH_REQUIRED", "true")
    monkeypatch.setenv("RWKV_ADMIN_PASSWORD_HASH", password_hash("rwkv", salt=b"0123456789abcdef"))
    monkeypatch.setenv("RWKV_ADMIN_SESSION_SECRET", "test-session-secret-that-is-longer-than-forty-three-characters")
    monkeypatch.setenv("RWKV_ADMIN_ALLOWED_ORIGINS", "https://testserver")
    monkeypatch.setenv("RWKV_ADMIN_SECURE_COOKIE", "true")
    headers = {"Origin": "https://testserver", "X-RWKV-Admin-Request": "1"}

    with TestClient(_app(), base_url="https://testserver") as client:
        assert client.get("/api/admin/auth/session").json()["authenticated"] is False
        assert client.get("/api/admin/protected").status_code == 401
        assert client.post("/api/admin/auth/login", json={"password": "rwkv"}).status_code == 403
        assert client.post(
            "/api/admin/auth/login",
            json={"password": "wrong"},
            headers=headers,
        ).status_code == 401

        login = client.post(
            "/api/admin/auth/login",
            json={"password": "rwkv"},
            headers=headers,
        )
        assert login.status_code == 200
        assert login.json()["authenticated"] is True
        cookie = login.headers["set-cookie"]
        assert "__Host-rwkv_admin_session=" in cookie
        assert "HttpOnly" in cookie
        assert "Secure" in cookie
        assert "SameSite=strict" in cookie
        assert "password" not in cookie.lower()
        assert "=rwkv;" not in cookie

        assert client.get("/api/admin/auth/session").json()["authenticated"] is True
        assert client.get("/api/admin/protected").json() == {"ok": True}
        assert client.post("/api/admin/protected").status_code == 403
        assert client.post("/api/admin/protected", headers=headers).json() == {"ok": True}

        logout = client.post("/api/admin/auth/logout", headers=headers)
        assert logout.status_code == 200
        assert logout.json()["authenticated"] is False
        assert client.get("/api/admin/protected").status_code == 401
