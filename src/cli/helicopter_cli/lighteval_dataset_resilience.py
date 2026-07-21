from __future__ import annotations

import functools
import logging
import os
import time
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
