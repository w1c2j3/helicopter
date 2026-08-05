"""Password authentication and signed sessions for the admin API."""

from __future__ import annotations

import asyncio
import base64
from collections import defaultdict, deque
from contextvars import ContextVar
import hashlib
import hmac
import json
import os
import secrets
from threading import Lock
import time
from typing import Deque

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse, Response


SESSION_COOKIE = "__Host-rwkv_admin_session"
_SESSION_CONTEXT: ContextVar[bool] = ContextVar("rwkv_admin_session", default=False)
_LOGIN_FAILURES: dict[str, Deque[float]] = defaultdict(deque)
_LOGIN_FAILURE_LOCK = Lock()
_LOGIN_WINDOW_SECONDS = 300
_LOGIN_MAX_FAILURES = 5


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def auth_required() -> bool:
    default = bool(
        (os.environ.get("RWKV_ADMIN_PASSWORD_HASH") or "").strip()
        or (os.environ.get("RWKV_ADMIN_API_KEY") or "").strip()
    )
    return _env_bool("RWKV_ADMIN_AUTH_REQUIRED", default)


def auth_configured() -> bool:
    return bool(
        (os.environ.get("RWKV_ADMIN_PASSWORD_HASH") or "").strip()
        and (os.environ.get("RWKV_ADMIN_SESSION_SECRET") or "").strip()
    )


def admin_session_active() -> bool:
    return _SESSION_CONTEXT.get()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def password_hash(password: str, *, iterations: int = 600_000, salt: bytes | None = None) -> str:
    resolved_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), resolved_salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64encode(resolved_salt)}${_b64encode(digest)}"


def verify_password(password: str) -> bool:
    if not password or len(password) > 256:
        return False
    encoded = (os.environ.get("RWKV_ADMIN_PASSWORD_HASH") or "").strip()
    try:
        algorithm, raw_iterations, raw_salt, expected_digest = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(raw_iterations)
        if iterations < 200_000 or iterations > 2_000_000:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _b64decode(raw_salt),
            iterations,
        )
        return hmac.compare_digest(_b64encode(digest), expected_digest)
    except (TypeError, ValueError):
        return False


def session_ttl_seconds() -> int:
    try:
        value = int(os.environ.get("RWKV_ADMIN_SESSION_TTL_SECONDS", "3600"))
    except ValueError:
        value = 3600
    return min(max(value, 300), 43_200)


def _session_secret() -> bytes:
    value = (os.environ.get("RWKV_ADMIN_SESSION_SECRET") or "").strip()
    if len(value) < 43:
        raise RuntimeError("RWKV_ADMIN_SESSION_SECRET must contain at least 256 bits")
    return value.encode("utf-8")


def create_session_token(*, now: int | None = None) -> tuple[str, int]:
    issued_at = int(time.time() if now is None else now)
    expires_at = issued_at + session_ttl_seconds()
    payload = {
        "v": 1,
        "iat": issued_at,
        "exp": expires_at,
        "nonce": secrets.token_urlsafe(18),
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _b64encode(hmac.new(_session_secret(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}", expires_at


def validate_session_token(token: str | None, *, now: int | None = None) -> dict[str, int | str] | None:
    if not token or len(token) > 2048 or not auth_configured():
        return None
    try:
        body, supplied_signature = token.split(".", 1)
        expected_signature = _b64encode(
            hmac.new(_session_secret(), body.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        payload = json.loads(_b64decode(body))
        current = int(time.time() if now is None else now)
        issued_at = int(payload["iat"])
        expires_at = int(payload["exp"])
        if payload.get("v") != 1 or issued_at > current + 30 or expires_at <= current:
            return None
        if expires_at - issued_at > session_ttl_seconds():
            return None
        return payload
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=session_ttl_seconds(),
        httponly=True,
        secure=_env_bool("RWKV_ADMIN_SECURE_COOKIE", True),
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        httponly=True,
        secure=_env_bool("RWKV_ADMIN_SECURE_COOKIE", True),
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"


def _allowed_origins() -> set[str]:
    return {
        item.strip().rstrip("/")
        for item in (os.environ.get("RWKV_ADMIN_ALLOWED_ORIGINS") or "").split(",")
        if item.strip()
    }


def validate_unsafe_request(request: Request) -> None:
    if request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = (request.headers.get("origin") or "").rstrip("/")
    allowed = _allowed_origins()
    if not origin or not allowed or origin not in allowed:
        raise HTTPException(status_code=403, detail="invalid admin request origin")
    if request.headers.get("x-rwkv-admin-request") != "1":
        raise HTTPException(status_code=403, detail="missing admin request proof")


def request_identifier(request: Request) -> str:
    forwarded = (request.headers.get("cf-connecting-ip") or "").strip()
    if forwarded:
        return forwarded[:128]
    client = request.client.host if request.client else "unknown"
    return client[:128]


def login_retry_after(identifier: str, *, now: float | None = None) -> int:
    current = time.monotonic() if now is None else now
    with _LOGIN_FAILURE_LOCK:
        failures = _LOGIN_FAILURES[identifier]
        while failures and current - failures[0] >= _LOGIN_WINDOW_SECONDS:
            failures.popleft()
        if len(failures) < _LOGIN_MAX_FAILURES:
            return 0
        return max(1, int(_LOGIN_WINDOW_SECONDS - (current - failures[0])))


def record_login_failure(identifier: str) -> None:
    with _LOGIN_FAILURE_LOCK:
        _LOGIN_FAILURES[identifier].append(time.monotonic())


def clear_login_failures(identifier: str) -> None:
    with _LOGIN_FAILURE_LOCK:
        _LOGIN_FAILURES.pop(identifier, None)


async def slow_failed_login() -> None:
    await asyncio.sleep(0.35 + secrets.randbelow(250) / 1000)


def install_admin_auth(app: FastAPI) -> None:
    @app.middleware("http")
    async def admin_auth_middleware(request: Request, call_next):
        path = request.url.path.rstrip("/")
        is_admin_api = path == "/api/admin" or path.startswith("/api/admin/")
        public_auth_paths = {
            "/api/admin/auth/login",
            "/api/admin/auth/logout",
            "/api/admin/auth/session",
        }
        if not is_admin_api or not auth_required() or path in public_auth_paths:
            return await call_next(request)
        if not auth_configured():
            return JSONResponse({"detail": "admin authentication is not configured"}, status_code=503)
        session = validate_session_token(request.cookies.get(SESSION_COOKIE))
        if session is None:
            return JSONResponse(
                {"detail": "authentication required"},
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        try:
            validate_unsafe_request(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        context_token = _SESSION_CONTEXT.set(True)
        try:
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            _SESSION_CONTEXT.reset(context_token)
