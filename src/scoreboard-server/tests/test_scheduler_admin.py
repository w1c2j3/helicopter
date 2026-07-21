from __future__ import annotations

import io
import time

import pytest

from scoreboard_server.cores import scheduler_admin
from scoreboard_server.cores.scheduler_admin import SchedulerAdminController, SchedulerAdminError


class _FakeProcess:
    pid = 43210

    def __init__(self, command: list[str], **_: object) -> None:
        self.command = command
        self.stdout = io.StringIO(
            "eval batch: [g1h-1.5b/lighteval] attempt 1 tasks=gsm8k|0\n"
            "eval batch: wrote report /tmp/report.json\n"
        )

    def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        return None


def test_scheduler_admin_builds_safe_cli_command_and_tracks_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    processes: list[_FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        process = _FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(scheduler_admin.subprocess, "Popen", fake_popen)
    controller = SchedulerAdminController()
    started = controller.start(
        {
            "config": "configs/local/g1h-1.5b-accuracy.toml",
            "models": ["g1h-1.5b"],
            "tasks": ["gsm8k|0"],
            "parallel": 2,
            "scoreboard": True,
        }
    )
    assert started["run_id"].startswith("admin-")
    assert len(processes) == 1
    command = processes[0].command
    assert command[1:5] == ["-m", "helicopter_cli", "eval", "batch"]
    assert command[command.index("--models") + 1] == "g1h-1.5b"
    assert command[command.index("--tasks") + 1] == "gsm8k|0"
    assert "--scoreboard" in command
    assert "|" not in command[:-1]

    for _ in range(50):
        status = controller.snapshot()
        if status["status"] == "completed":
            break
        time.sleep(0.01)
    assert status["status"] == "completed"
    assert status["completed_jobs"] == 1
    assert status["log_tail"]


def test_scheduler_admin_rejects_conflicting_task_sources() -> None:
    controller = SchedulerAdminController()
    with pytest.raises(SchedulerAdminError, match="不能同时设置"):
        controller._normalize({"tasks": ["gsm8k|0"], "tasks_from_db": True})
