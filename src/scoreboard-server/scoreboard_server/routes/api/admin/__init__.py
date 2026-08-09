from __future__ import annotations

from fastapi import FastAPI

from . import auth, backpressure, eval, health


def register(app: FastAPI) -> None:
    auth.register(app)
    health.register(app)
    eval.register(app)
    backpressure.register(app)
