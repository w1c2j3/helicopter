from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from helicopter_lm_eval import __main__ as entrypoint
from helicopter_lm_eval import native
from helicopter_lm_eval.route import RouteError, read_toml, select_route


ROOT = Path(__file__).resolve().parents[1]


def test_route_selects_rwkv_only_for_explicit_vllm_http(tmp_path: Path) -> None:
    rwkv = tmp_path / "rwkv.toml"
    rwkv.write_text('backend = "vllm_http"\n', encoding="utf-8")
    native_config = tmp_path / "native.toml"
    native_config.write_text('model = "hf"\n', encoding="utf-8")

    assert select_route(read_toml(rwkv)) == "rwkv"
    assert select_route(read_toml(native_config)) == "native"


def test_route_rejects_other_backend_values() -> None:
    with pytest.raises(RouteError, match="omitted for native lm-eval"):
        select_route({"backend": "hf"})


def test_entrypoint_dispatches_native_without_calling_rwkv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "native.toml"
    config.write_text('model = "hf"\ntasks = ["hellaswag"]\n', encoding="utf-8")
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        entrypoint,
        "run_native",
        lambda **kwargs: calls.append(("native", kwargs)) or 0,
    )
    monkeypatch.setattr(
        entrypoint,
        "run_rwkv",
        lambda **kwargs: calls.append(("rwkv", kwargs)) or 0,
    )

    assert entrypoint.main(["--config", str(config), "--dry-run"]) == 0
    assert [name for name, _kwargs in calls] == ["native"]
    assert calls[0][1]["dry_run"] is True


def test_entrypoint_dispatches_explicit_backend_to_rwkv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "rwkv.toml"
    config.write_text('backend = "vllm_http"\n', encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        entrypoint,
        "run_native",
        lambda **_kwargs: calls.append("native") or 0,
    )
    monkeypatch.setattr(
        entrypoint,
        "run_rwkv",
        lambda **_kwargs: calls.append("rwkv") or 0,
    )

    assert entrypoint.main(["--config", str(config)]) == 0
    assert calls == ["rwkv"]


def test_native_run_delegates_complete_config_to_lm_eval_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []

    def run(command, *, check):
        config_index = command.index("--config") + 1
        import yaml

        loaded = yaml.safe_load(Path(command[config_index]).read_text(encoding="utf-8"))
        assert loaded == {
            "model": "hf",
            "model_args": {"pretrained": "Qwen/Qwen3.5-0.8B-Base"},
            "tasks": ["hellaswag"],
            "output_path": ".tmp/eval/qwen",
            "log_samples": True,
        }
        calls.append((command, check))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(native.subprocess, "run", run)

    assert native.run(
        config={
            "model": "hf",
            "model_args": {"pretrained": "Qwen/Qwen3.5-0.8B-Base"},
            "tasks": ["hellaswag"],
            "output_path": ".tmp/eval/qwen",
            "log_samples": True,
        },
        dry_run=False,
    ) == 0
    assert calls[0][0][1:4] == ["-m", "lm_eval", "run"]
    assert calls[0][1] is False


def test_native_dry_run_resolves_upstream_tasks_without_loading_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert native.run(
        config={
            "model": "hf",
            "model_args": {"pretrained": "gpt2"},
            "tasks": ["hellaswag"],
            "output_path": ".tmp/eval/native",
            "log_samples": True,
        },
        dry_run=True,
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["route"] == "native"
    assert output["model"] == "hf"
    assert output["resolved_tasks"] == ["hellaswag"]


def test_native_dry_run_rejects_unknown_task_selector() -> None:
    with pytest.raises(native.NativeConfigError, match="Tasks not found"):
        native.run(
            config={
                "model": "hf",
                "tasks": ["definitely_not_an_lm_eval_task"],
            },
            dry_run=True,
        )


def test_launcher_skips_rwkv_service_for_native_config(tmp_path: Path) -> None:
    config = tmp_path / "native.toml"
    config.write_text('model = "hf"\ntasks = ["hellaswag"]\n', encoding="utf-8")
    fake_cli = tmp_path / "helicopter"
    fake_cli.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o700)
    env = os.environ.copy()
    env["HELICOPTER_CLI"] = str(fake_cli)
    env["HELICOPTER_LM_EVAL_PYTHON"] = sys.executable

    completed = subprocess.run(
        ["bash", "scripts/run_lm_eval.sh", str(config), "--dry-run"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "eval",
        "--evaluator",
        "lm-eval",
        "--config",
        str(config),
        "--dry-run",
    ]


@pytest.mark.parametrize(
    "field",
    ["benchmark_configs", "pool_manifests", "prompt", "weights", "wkv_modes"],
)
def test_native_config_rejects_rwkv_only_fields(field: str) -> None:
    with pytest.raises(native.NativeConfigError, match=field):
        native.run(
            config={
                "model": "hf",
                "tasks": ["hellaswag"],
                field: [],
            },
            dry_run=True,
        )
