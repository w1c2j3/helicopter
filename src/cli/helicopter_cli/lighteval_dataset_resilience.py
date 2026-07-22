from __future__ import annotations

import functools
import importlib
import logging
import os
import time
from contextlib import contextmanager
from collections.abc import Callable
from typing import Any


LOGGER = logging.getLogger(__name__)
_PATCH_MARKER = "_helicopter_dataset_resilience_patched"


def _exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_dataset_error(error: BaseException) -> bool:
    """Return whether a dataset failure is likely caused by a short network fault."""
    transient_names = {
        "ChunkedEncodingError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionResetError",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "SSLError",
        "TimeoutError",
    }
    return any(type(item).__name__ in transient_names for item in _exception_chain(error))


def is_offline_cache_miss(error: BaseException) -> bool:
    messages = " ".join(str(item) for item in _exception_chain(error)).lower()
    return (
        "couldn't find cache" in messages
        or "offlinemodeisenabled" in messages
        or "offline mode is enabled" in messages
        or "offline mode" in messages and "cache" in messages
    )


def _online_fallback_enabled() -> bool:
    return os.environ.get(
        "HELICOPTER_LIGHTEVAL_DATASET_ONLINE_FALLBACK",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _reset_huggingface_sessions() -> None:
    try:
        http = importlib.import_module("huggingface_hub.utils._http")
        reset_sessions = getattr(http, "reset_sessions")
    except (AttributeError, ImportError):
        return
    reset_sessions()


@contextmanager
def _temporary_online_dataset_access():
    env_names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    saved_env = {name: os.environ.get(name) for name in env_names}
    saved_attributes: list[tuple[Any, str, Any]] = []
    for name in env_names:
        os.environ[name] = "0"
    for module_name, attribute in (
        ("datasets.config", "HF_DATASETS_OFFLINE"),
        ("datasets.config", "HF_HUB_OFFLINE"),
        ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
    ):
        try:
            module = importlib.import_module(module_name)
            saved_attributes.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, False)
        except (AttributeError, ImportError):
            continue
    _reset_huggingface_sessions()
    try:
        yield
    finally:
        for module, attribute, value in reversed(saved_attributes):
            setattr(module, attribute, value)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _reset_huggingface_sessions()


def _retry_count() -> int:
    value = os.environ.get("HELICOPTER_LIGHTEVAL_DATASET_RETRIES", "2")
    try:
        return max(0, int(value))
    except ValueError:
        return 2


def _retry_delay() -> float:
    value = os.environ.get("HELICOPTER_LIGHTEVAL_DATASET_RETRY_DELAY", "1.0")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 1.0


def retry_load_dataset(load_dataset: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Retry only transient failures while preserving normal dataset errors."""
    retries = _retry_count()
    delay = _retry_delay()
    for attempt in range(retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as error:
            if is_offline_cache_miss(error) and _online_fallback_enabled():
                LOGGER.warning(
                    "LightEval dataset is absent from the offline cache; "
                    "enabling network access for this dataset only"
                )
                with _temporary_online_dataset_access():
                    for online_attempt in range(retries + 1):
                        try:
                            return load_dataset(*args, **kwargs)
                        except Exception as online_error:
                            if (
                                online_attempt >= retries
                                or not is_transient_dataset_error(online_error)
                            ):
                                raise
                            LOGGER.warning(
                                "LightEval dataset download failed (%s/%s); "
                                "retrying in %.1fs: %s",
                                online_attempt + 1,
                                retries + 1,
                                delay,
                                online_error,
                            )
                            if delay:
                                time.sleep(delay * (online_attempt + 1))
            if attempt >= retries or not is_transient_dataset_error(error):
                raise
            LOGGER.warning(
                "LightEval dataset download failed (%s/%s); retrying in %.1fs: %s",
                attempt + 1,
                retries + 1,
                delay,
                error,
            )
            if delay:
                time.sleep(delay * (attempt + 1))
    raise AssertionError("unreachable")


def patch_lighteval_dataset_loading() -> None:
    from lighteval.tasks import lighteval_task

    original = lighteval_task.load_dataset
    if getattr(original, _PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def resilient_load_dataset(*args: Any, **kwargs: Any) -> Any:
        return retry_load_dataset(original, *args, **kwargs)

    setattr(resilient_load_dataset, _PATCH_MARKER, True)
    lighteval_task.load_dataset = resilient_load_dataset


patch_lighteval_dataset_loading()
