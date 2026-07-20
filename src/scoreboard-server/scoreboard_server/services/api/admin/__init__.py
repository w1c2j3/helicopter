"""Use-case services shared by the admin API route leaves."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException


def auth_required() -> bool:
    return bool((os.environ.get("RWKV_ADMIN_API_KEY") or "").strip())


def check_admin_auth(authorization: str | None) -> None:
    expected = (os.environ.get("RWKV_ADMIN_API_KEY") or "").strip()
    if not expected:
        return
    supplied = authorization or ""
    if hmac.compare_digest(supplied, f"Bearer {expected}"):
        return
    raise HTTPException(status_code=401, detail="unauthorized")
