from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from scoreboard_server.services.api.admin.auth import (
    SESSION_COOKIE,
    auth_configured,
    auth_required,
    clear_login_failures,
    clear_session_cookie,
    create_session_token,
    login_retry_after,
    record_login_failure,
    request_identifier,
    set_session_cookie,
    slow_failed_login,
    validate_session_token,
    validate_unsafe_request,
    verify_password,
)


class AdminLoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def register(app: FastAPI) -> None:
    @app.post("/api/admin/auth/login")
    async def admin_login(request: Request, payload: AdminLoginRequest = Body(...)) -> JSONResponse:
        validate_unsafe_request(request)
        if not auth_required() or not auth_configured():
            raise HTTPException(status_code=503, detail="admin authentication is not configured")
        identifier = request_identifier(request)
        retry_after = login_retry_after(identifier)
        if retry_after:
            raise HTTPException(
                status_code=429,
                detail="too many login attempts",
                headers={"Retry-After": str(retry_after)},
            )
        if not verify_password(payload.password):
            record_login_failure(identifier)
            await slow_failed_login()
            raise HTTPException(status_code=401, detail="invalid credentials")
        clear_login_failures(identifier)
        token, expires_at = create_session_token()
        response = JSONResponse(
            {
                "authenticated": True,
                "expires_at": expires_at,
            }
        )
        set_session_cookie(response, token)
        return response

    @app.get("/api/admin/auth/session")
    async def admin_session(request: Request) -> JSONResponse:
        session = validate_session_token(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse(
            {
                "authenticated": session is not None,
                "expires_at": int(session["exp"]) if session is not None else None,
            }
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/admin/auth/logout")
    async def admin_logout(request: Request) -> JSONResponse:
        validate_unsafe_request(request)
        response = JSONResponse({"authenticated": False, "expires_at": None})
        clear_session_cookie(response)
        return response
