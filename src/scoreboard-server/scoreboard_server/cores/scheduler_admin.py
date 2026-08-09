from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = REPO_ROOT / "results" / "admin"
TERMINAL_STATES = {"idle", "completed", "cancelled", "failed"}
_START_LINE = re.compile(r"eval batch: \[(?P<job>[^]]+)] attempt")


class SchedulerAdminError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise SchedulerAdminError("models/tasks 必须是字符串或字符串数组")


def _job_list(value: Any) -> list[str]:
    """Keep commas inside MODEL=BENCHMARK[,BENCHMARK...] job values."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise SchedulerAdminError("jobs 必须是字符串或字符串数组")


def _redacted_request(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("api_key", "infer_api_key"):
        if result.get(key):
            result[key] = "***"
    return result


def _nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _nested_value(child, key)
            if found is not None:
                return found
    return None


def _prometheus_value(text: str, metric: str) -> int:
    total = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(metric):
            continue
        try:
            total += float(stripped.rsplit(None, 1)[-1])
        except (IndexError, ValueError):
            continue
    return max(0, int(total))


def _latest_config() -> str:
    local = REPO_ROOT / "configs" / "local"
    candidates = sorted(local.glob("*.toml"), key=lambda item: item.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].relative_to(REPO_ROOT).as_posix()
    return "configs/example.toml"


def _gpu_options() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, TypeError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    result: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        result.append(
            {
                "id": parts[0],
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_used_mib": int(float(parts[3])),
            }
        )
    return result


def admin_options(*, include_gpu: bool = True) -> dict[str, Any]:
    configs = sorted((REPO_ROOT / "configs").rglob("*.toml"))
    models: set[str] = set()
    model_options: dict[str, dict[str, Any]] = {}
    tasks: set[str] = set()
    for path in configs:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        config_name = path.relative_to(REPO_ROOT).as_posix()
        for key, raw in (data.get("models") or {}).items():
            if not isinstance(raw, dict) or key == "deployed":
                continue
            name = str(key)
            models.add(name)
            option = model_options.setdefault(
                name,
                {
                    "name": name,
                    "weight_path": raw.get("path") or raw.get("file"),
                    "served_model_name": raw.get("served_model_name"),
                    "configs": [],
                    "runtime": "local-vllm-rwkv",
                },
            )
            option["configs"].append(config_name)
            if not option.get("weight_path"):
                option["weight_path"] = raw.get("path") or raw.get("file")
            if not option.get("served_model_name"):
                option["served_model_name"] = raw.get("served_model_name")
        light = data.get("lighteval") or {}
        batch = (data.get("eval") or {}).get("batch") or {}
        tasks.update(_as_list(batch.get("tasks") or light.get("tasks")))
    return {
        "jobs": [{"name": task, "domain": "lighteval"} for task in sorted(tasks)],
        "domains": sorted(tasks),
        "model_select": sorted(models),
        "model_options": [model_options[name] for name in sorted(model_options)],
        "gpu_options": _gpu_options() if include_gpu else [],
        "worker_profile": ["local-managed", "existing-endpoint"],
        "protocol": ["openai"],
        "run_mode": ["skip-completed", "rerun"],
        "configs": [path.relative_to(REPO_ROOT).as_posix() for path in configs],
    }


def admin_draft() -> dict[str, Any]:
    # Starting a run only needs config defaults; avoid probing hardware on the
    # process-launch path. The options endpoint can still expose GPU details.
    options = admin_options(include_gpu=False)
    config = _latest_config()
    config_models: list[str] = []
    config_tasks: list[str] = []
    try:
        data = tomllib.loads((REPO_ROOT / config).read_text(encoding="utf-8"))
        config_models = [str(key) for key in (data.get("models") or {})]
        light = data.get("lighteval") or {}
        batch = (data.get("eval") or {}).get("batch") or {}
        config_tasks = _as_list(batch.get("tasks") or light.get("tasks"))
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return {
        "config": config,
        "jobs": [],
        "models": config_models[:1] or options["model_select"][:1],
        "tasks": config_tasks[:1] or options["domains"][:1],
        "fc_tasks": [],
        "gpus": "",
        "parallel": None,
        "max_retries": 0,
        "no_server": False,
        "scoreboard": True,
        "rerun": False,
        "dry_run": False,
    }


@dataclass
class RunState:
    run_id: str
    request: dict[str, Any]
    private_request: dict[str, Any] = field(repr=False)
    command: list[str]
    report_path: Path
    log_path: Path
    process: subprocess.Popen[str]
    status: str = "starting"
    desired_state: str | None = None
    error: str | None = None
    started_at_unix_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_unix_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    finished_at_unix_ms: int | None = None
    planned_jobs: list[str] = field(default_factory=list)
    active_jobs: list[str] = field(default_factory=list)
    completed_jobs: int = 0
    failed_jobs: int = 0
    available_gpus: list[str] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)


class SchedulerAdminController:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: RunState | None = None

    def draft(self) -> dict[str, Any]:
        return admin_draft()

    def options(self) -> dict[str, Any]:
        return admin_options()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._state and self._state.status not in TERMINAL_STATES:
                raise SchedulerAdminError("已有评测任务正在运行", 409)
            request = self._normalize(payload)
            run_id = f"admin-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
            report_path = RESULTS_ROOT / f"{run_id}.json"
            log_path = RESULTS_ROOT / f"{run_id}.log"
            command = self._command(request, report_path)
            env = os.environ.copy()
            cli_path = str(REPO_ROOT / "src" / "cli")
            env["PYTHONPATH"] = cli_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            kwargs: dict[str, Any] = {
                "cwd": REPO_ROOT,
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "posix":
                kwargs["start_new_session"] = True
            elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                process = subprocess.Popen(command, **kwargs)
            except OSError as exc:
                raise SchedulerAdminError(f"无法启动评测进程：{exc}", 503) from exc
            models = request["models"] or ["config-models"]
            jobs = [str(job) for job in request.get("jobs") or []]
            if not jobs:
                for model in models:
                    if request["tasks"] or request["tasks_from_db"]:
                        jobs.append(f"{model}/lighteval")
                    if request["fc_tasks"]:
                        jobs.append(f"{model}/function-calling")
            if not jobs:
                jobs = ["config/batch"]
            gpus = [item.strip() for item in str(request.get("gpus") or "").split(",") if item.strip()]
            state = RunState(
                run_id=run_id,
                request=_redacted_request(request),
                private_request=request,
                command=command,
                report_path=report_path,
                log_path=log_path,
                process=process,
                status="running",
                planned_jobs=jobs,
                available_gpus=gpus,
            )
            self._state = state
            threading.Thread(target=self._consume, args=(state,), daemon=True, name=f"{run_id}-log").start()
            return self.snapshot()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            state = self._require_active()
            if os.name != "posix":
                raise SchedulerAdminError("当前平台不支持安全暂停；可以取消任务", 409)
            os.killpg(state.process.pid, signal.SIGSTOP)
            state.status = "paused"
            state.desired_state = "paused"
            state.updated_at_unix_ms = int(time.time() * 1000)
            return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            state = self._require_active(allow_paused=True)
            if state.status != "paused":
                raise SchedulerAdminError("当前任务未暂停", 409)
            if os.name != "posix":
                raise SchedulerAdminError("当前平台不支持恢复暂停任务", 409)
            os.killpg(state.process.pid, signal.SIGCONT)
            state.status = "running"
            state.desired_state = "running"
            state.updated_at_unix_ms = int(time.time() * 1000)
            return self.snapshot()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            state = self._require_active(allow_paused=True)
            state.desired_state = "cancelled"
            if state.status == "paused" and os.name == "posix":
                os.killpg(state.process.pid, signal.SIGCONT)
            if os.name == "posix":
                os.killpg(state.process.pid, signal.SIGTERM)
            else:
                state.process.terminate()
            state.status = "cancelling"
            state.updated_at_unix_ms = int(time.time() * 1000)
            threading.Thread(target=self._kill_after_grace, args=(state,), daemon=True).start()
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            if state is None:
                return {
                    "status": "idle", "desired_state": None, "run_id": None, "error": None,
                    "started_at_unix_ms": None, "updated_at_unix_ms": None, "finished_at_unix_ms": None,
                    "pending_jobs": 0, "running_jobs": 0, "completed_jobs": 0, "failed_jobs": 0,
                    "tasks_total": 0, "progress_percent": 0.0, "queue_head": [], "active_jobs": [],
                    "available_gpus": [], "request": None, "log_path": None, "report_path": None, "log_tail": [],
                }
            total = max(len(state.planned_jobs), state.completed_jobs + state.failed_jobs)
            finished = state.completed_jobs + state.failed_jobs
            active = list(state.active_jobs)
            pending = max(total - finished - len(active), 0)
            return {
                "status": state.status,
                "desired_state": state.desired_state,
                "run_id": state.run_id,
                "error": state.error,
                "started_at_unix_ms": state.started_at_unix_ms,
                "updated_at_unix_ms": state.updated_at_unix_ms,
                "finished_at_unix_ms": state.finished_at_unix_ms,
                "pending_jobs": pending,
                "running_jobs": len(active) if state.status in {"running", "starting"} else 0,
                "completed_jobs": state.completed_jobs,
                "failed_jobs": state.failed_jobs,
                "tasks_total": total,
                "progress_percent": (finished / total) if total else 0.0,
                "queue_head": [job for job in state.planned_jobs if job not in active][:12],
                "active_jobs": active,
                "available_gpus": state.available_gpus,
                "request": state.request,
                "log_path": str(state.log_path),
                "report_path": str(state.report_path),
                "log_tail": list(state.log_tail[-30:]),
            }

    def backpressure(self, infer_base_url: str | None = None) -> dict[str, Any]:
        snapshot = self.snapshot()
        with self._lock:
            request = dict(self._state.private_request) if self._state is not None else {}
        base_url = str(infer_base_url or request.get("base_url") or "").strip()
        models: list[dict[str, Any]] = []
        error: str | None = None
        if base_url:
            root = base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[:-3]
            headers = {"Accept": "application/json"}
            raw_key = request.get("api_key")
            if raw_key and raw_key != "***":
                headers["Authorization"] = f"Bearer {raw_key}"
            try:
                info_url = f"{root}/server_info?config_format=json"
                with urllib.request.urlopen(
                    urllib.request.Request(info_url, headers=headers), timeout=3.0
                ) as response:
                    server_info = json.loads(response.read().decode("utf-8"))
                metrics_url = f"{root}/metrics"
                with urllib.request.urlopen(
                    urllib.request.Request(metrics_url, headers=headers), timeout=3.0
                ) as response:
                    metrics = response.read().decode("utf-8", errors="replace")
                requested_models = request.get("models") or ["served-model"]
                models = [
                    {
                        "model_slug": str(model),
                        "max_num_seqs": _nested_value(server_info, "max_num_seqs"),
                        "max_num_batched_tokens": _nested_value(
                            server_info, "max_num_batched_tokens"
                        ),
                        "num_requests_running": _prometheus_value(
                            metrics, "vllm:num_requests_running"
                        ),
                        "num_requests_waiting": _prometheus_value(
                            metrics, "vllm:num_requests_waiting"
                        ),
                        "source": "vllm:/server_info+/metrics",
                    }
                    for model in requested_models
                ]
            except (OSError, ValueError, urllib.error.URLError) as exc:
                error = str(exc)
        return {"infer_base_url": base_url, "available_gpus": snapshot["available_gpus"], "models": models, "error": error}

    def _kill_after_grace(self, state: RunState) -> None:
        try:
            state.process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(state.process.pid, signal.SIGKILL)
            else:
                state.process.kill()
        except ProcessLookupError:
            pass

    def _require_active(self, *, allow_paused: bool = False) -> RunState:
        state = self._state
        valid = {"running", "starting", "cancelling"}
        if allow_paused:
            valid.add("paused")
        if state is None or state.status not in valid:
            raise SchedulerAdminError("当前没有可控制的评测任务", 409)
        return state

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(admin_draft())
        result.update(payload or {})
        incoming = payload or {}
        result["jobs"] = _job_list(result.get("jobs"))
        if result["jobs"] and any(
            incoming.get(key)
            for key in ("models", "model_select", "tasks", "domains", "fc_tasks", "tasks_from_db")
        ):
            raise SchedulerAdminError("jobs 不能与 models/tasks/fc_tasks 同时设置")
        if result["jobs"]:
            result["models"] = []
            result["tasks"] = []
            result["fc_tasks"] = []
            result["tasks_from_db"] = False
        result["models"] = _as_list(result.get("models") or result.get("model_select"))
        result["tasks"] = _as_list(result.get("tasks") or result.get("domains"))
        result["fc_tasks"] = _as_list(result.get("fc_tasks"))
        result["benchmark_fields"] = _as_list(result.get("benchmark_fields"))
        if result.get("parallel") not in (None, ""):
            result["parallel"] = max(1, int(result["parallel"]))
        else:
            result["parallel"] = None
        result["max_retries"] = max(0, int(result.get("max_retries") or 0))
        result["tasks_from_db"] = bool(result.get("tasks_from_db"))
        if result["tasks_from_db"] and result["tasks"]:
            raise SchedulerAdminError("tasks_from_db 与 tasks 不能同时设置")
        return result

    def _command(self, request: dict[str, Any], report_path: Path) -> list[str]:
        python = os.environ.get("HELICOPTER_PYTHON")
        if not python:
            candidate = REPO_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python = str(candidate if candidate.exists() else Path(sys.executable))
        command = [python, "-m", "helicopter_cli", "eval", "batch"]
        value_flags = {
            "config": "--config", "env_file": "--env-file", "gpus": "--gpus", "parallel": "--parallel",
            "max_retries": "--max-retries", "port_base": "--port-base", "base_url": "--base-url",
            "backend": "--backend", "api_key": "--api-key", "concurrent_requests": "--concurrent-requests",
            "max_tokens": "--max-tokens", "max_samples": "--max-samples", "benchmark_scope": "--benchmark-scope",
            "benchmark_limit": "--benchmark-limit",
        }
        for key, flag in value_flags.items():
            value = request.get(key)
            if value not in (None, "", []):
                command.extend([flag, str(value)])
        for key, flag in (("models", "--models"), ("tasks", "--tasks"), ("fc_tasks", "--fc-tasks"), ("benchmark_fields", "--benchmark-fields")):
            for value in request.get(key) or []:
                command.extend([flag, str(value)])
        for value in request.get("jobs") or []:
            command.extend(["--job", str(value)])
        for key, flag in (("tasks_from_db", "--tasks-from-db"), ("rerun", "--rerun"), ("no_server", "--no-server"), ("scoreboard", "--scoreboard"), ("dry_run", "--dry-run")):
            if request.get(key):
                command.append(flag)
        command.extend(["--batch-output", str(report_path)])
        return command

    def _consume(self, state: RunState) -> None:
        try:
            with state.log_path.open("a", encoding="utf-8") as log:
                assert state.process.stdout is not None
                for line in state.process.stdout:
                    log.write(line)
                    log.flush()
                    text = line.rstrip()
                    with self._lock:
                        state.log_tail.append(text)
                        state.log_tail = state.log_tail[-60:]
                        match = _START_LINE.search(text)
                        if match:
                            state.active_jobs = [match.group("job")]
                        state.updated_at_unix_ms = int(time.time() * 1000)
            exit_code = state.process.wait()
        except Exception as exc:  # noqa: BLE001
            exit_code = -1
            with self._lock:
                state.error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            state.active_jobs = []
            state.finished_at_unix_ms = int(time.time() * 1000)
            state.updated_at_unix_ms = state.finished_at_unix_ms
            if state.desired_state == "cancelled":
                state.status = "cancelled"
            elif exit_code == 0:
                state.status = "completed"
            else:
                state.status = "failed"
                if not state.error:
                    state.error = f"评测进程退出码 {exit_code}"
            self._apply_report(state)

    def _apply_report(self, state: RunState) -> None:
        if not state.report_path.is_file():
            if state.status == "completed":
                state.completed_jobs = len(state.planned_jobs)
            elif state.status == "failed":
                state.failed_jobs = max(1, len(state.planned_jobs))
            return
        try:
            payload = json.loads(state.report_path.read_text(encoding="utf-8"))
            units = payload.get("units") or []
            state.completed_jobs = sum(1 for unit in units if unit.get("status") in {"completed", "skipped", "dry_run"})
            state.failed_jobs = sum(1 for unit in units if unit.get("status") == "failed")
            if units:
                state.planned_jobs = [f"{unit.get('model', '?')}/{unit.get('kind', '?')}" for unit in units]
        except (OSError, ValueError, TypeError) as exc:
            state.error = state.error or f"无法读取批处理报告：{exc}"


controller = SchedulerAdminController()
