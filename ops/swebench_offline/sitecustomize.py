"""Serve SWE-bench raw GitHub files from the local bare-repository cache.

This module is loaded through ``PYTHONPATH`` for offline EvalScope runs.  It
only intercepts GET requests to ``raw.githubusercontent.com`` when the exact
owner/repository/commit/path tuple exists in the configured local cache.  All
other requests keep the normal ``requests`` behaviour.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


_CACHE_ROOT = Path(
    os.environ.get("SWE_BENCH_REPO_CACHE", "/home/rwkv/chase/swe_repo_cache")
)
_ORIGINAL_REQUEST = requests.sessions.Session.request
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _bare_repo_candidates(owner: str, repo: str) -> tuple[Path, ...]:
    repo_dir = f"repo__{owner}__{repo}.git"
    return (
        _CACHE_ROOT / "verified" / repo_dir,
        _CACHE_ROOT / "pro" / repo_dir,
        _CACHE_ROOT / "multilingual" / repo_dir,
        _CACHE_ROOT / repo_dir,
    )


def _cached_raw_file(url: str) -> bytes | None:
    parsed = urlparse(url)
    if parsed.hostname != "raw.githubusercontent.com":
        return None
    parts = [unquote(part) for part in parsed.path.lstrip("/").split("/", 3)]
    if len(parts) != 4:
        return None
    owner, repo, commit, file_path = parts
    if not all(_SAFE_COMPONENT.fullmatch(value) for value in (owner, repo, commit)):
        return None
    if not file_path or any(part in {"", ".", ".."} for part in file_path.split("/")):
        return None

    revision = f"{commit}:{file_path}"
    for bare_repo in _bare_repo_candidates(owner, repo):
        if not bare_repo.is_dir():
            continue
        completed = subprocess.run(
            ["git", f"--git-dir={bare_repo}", "show", revision],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return completed.stdout
    return None


def _offline_request(self, method, url, *args, **kwargs):  # noqa: ANN001, ANN202
    if str(method).upper() == "GET":
        content = _cached_raw_file(str(url))
        if content is not None:
            response = requests.Response()
            response.status_code = 200
            response.url = str(url)
            response.headers["Content-Type"] = "text/plain; charset=utf-8"
            response._content = content
            response.encoding = "utf-8"
            return response
    return _ORIGINAL_REQUEST(self, method, url, *args, **kwargs)


requests.sessions.Session.request = _offline_request
