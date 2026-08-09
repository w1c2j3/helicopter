"""Use-case services shared by the admin API route leaves."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException

from .auth import admin_session_active, auth_required


def check_admin_auth(authorization: str | None) -> None:
    if not auth_required():
        return
    if admin_session_active():
        return
    # Retain the legacy bearer key only for non-session deployments. The
    # public deployment configures password sessions and never sends this key
    # to the browser.
    expected = (os.environ.get("RWKV_ADMIN_API_KEY") or "").strip()
    supplied = authorization or ""
    if expected and hmac.compare_digest(supplied, f"Bearer {expected}"):
        return
    raise HTTPException(status_code=401, detail="unauthorized")
