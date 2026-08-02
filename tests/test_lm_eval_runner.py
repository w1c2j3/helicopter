from __future__ import annotations

import json
import stat
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from helicopter_lm_eval import evaluate
from helicopter_lm_eval.config import ConfigError, LMEvalConfig


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 10,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": 10240,
                "replicas": [
                    {
                        "base_url": "http://127.0.0.1:8000",
                        "max_concurrency": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_task_resolution_accepts_common_request_types_and_groups() -> None:
    manager, resolved = evaluate._resolve_tasks(
        ("wikitext", "hellaswag", "gsm8k", "mmlu")
    )

    assert resolved == ("wikitext", "hellaswag", "gsm8k", "mmlu")
    assert manager.task_index["wikitext"].cfg["output_type"] == (
        "loglikelihood_rolling"
    )
    assert manager.task_index["hellaswag"].cfg["output_type"] == "multiple_choice"
    assert manager.task_index["gsm8k"].cfg["output_type"] == "generate_until"
    assert manager.task_index["mmlu"].cfg["group"] == "mmlu"


def test_task_resolution_supports_globs_and_rejects_unknown_selectors() -> None:
    _manager, resolved = evaluate._resolve_tasks(("arc_easy", "gsm8k*"))

    assert resolved[0] == "arc_easy"
    assert "gsm8k" in resolved
    assert "gsm8k_cot" in resolved
    with pytest.raises(ConfigError, match="unknown lm-eval task, group, tag, or pattern"):
        evaluate._resolve_tasks(("does-not-exist",))


def test_qwen35_alignment_suite_uses_stable_unlimited_selectors() -> None:
    with (ROOT / "configs/eval/lm_eval_qwen35.toml").open("rb") as stream:
        config = tomllib.load(stream)

    expected = (
        "mmlu_pro",
        "mmlu_redux_generative",
        "ceval-valid",
        "gpqa_diamond_cot_zeroshot",
        "ifeval",
        "mmmlu",
    )
    manager, resolved = evaluate._resolve_tasks(tuple(config["tasks"]))

    assert resolved == expected
    assert "limit" not in config
    assert manager.task_index["mmlu_pro"].cfg["group"] == "mmlu_pro"
    assert manager.task_index["mmlu_redux_generative"].cfg["group"] == (
        "mmlu_redux_generative"
    )
    assert manager.task_index["ceval-valid"].cfg["group"] == "ceval-valid"
    assert manager.task_index["gpqa_diamond_cot_zeroshot"].cfg["output_type"] == (
        "generate_until"
    )
    assert manager.task_index["ifeval"].cfg["output_type"] == "generate_until"
    assert manager.task_index["mmmlu"].cfg["group"] == "mmmlu"


def test_dry_run_preflights_without_starting_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    config = tmp_path / "eval.toml"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
            ]
        ),
        encoding="utf-8",
    )
    closed: list[bool] = []

    class Pool:
        def __init__(self, configured_manifest):
            self.manifest = configured_manifest

        def preflight(self):
            return "rwkv-current"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(evaluate, "VLLMHttpPool", Pool)

    assert (
        evaluate.run(
            config_path=config,
            env={"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
            dry_run=True,
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"
    assert output["model_id"] == "rwkv-current"
    assert output["effective_max_length"] == 10238
    assert output["resolved_tasks"] == ["wikitext"]
    assert closed == [True]


def test_run_dispatches_choice_and_generation_tasks_to_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lm_eval

    manifest = _write_manifest(tmp_path)
    output_dir = tmp_path / "results"
    config = tmp_path / "eval.toml"
    config.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["hellaswag", "gsm8k"]',
                f'output_dir = "{output_dir}"',
                "batch_size = 4",
                "max_gen_toks = 512",
                "limit = 1",
            ]
        ),
        encoding="utf-8",
    )
    closed: list[bool] = []
    calls: list[dict[str, object]] = []

    class Pool:
        total_capacity = 2

        def __init__(self, configured_manifest):
            self.manifest = configured_manifest
            self.model_id = "rwkv-current"

        def preflight(self):
            return self.model_id

        def close(self):
            closed.append(True)

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "results": {
                "hellaswag": {"acc,none": 0.5},
                "gsm8k": {"exact_match,strict-match": 0.25},
            },
            "versions": {"hellaswag": 1, "gsm8k": 3},
        }

    monkeypatch.setattr(evaluate, "VLLMHttpPool", Pool)
    monkeypatch.setattr(lm_eval, "simple_evaluate", simple_evaluate)

    assert (
        evaluate.run(
            config_path=config,
            env={"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
            dry_run=False,
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["tasks"] == ["hellaswag", "gsm8k"]
    assert calls[0]["batch_size"] == 4
    assert calls[0]["limit"] == 1
    model = calls[0]["model"]
    assert model.max_gen_toks == 512
    assert json.loads((output_dir / "summary.json").read_text())["tasks"] == [
        "hellaswag",
        "gsm8k",
    ]
    assert closed == [True]


def test_result_writer_preserves_raw_results_and_normalizes_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir(mode=0o755)
    output_dir.chmod(0o755)
    config = LMEvalConfig(
        tasks=("wikitext",),
        output_dir=output_dir,
        batch_size=1,
        eot_token_id=0,
        max_gen_toks=256,
        limit=None,
        log_samples=False,
        vllm_pool_manifest=tmp_path / "pool.json",
        manifest=SimpleNamespace(
            global_step=10,
            wkv_mode="fp32io16",
            max_model_len=10240,
        ),
    )
    raw = {
        "results": {"wikitext": {"word_perplexity,none": 12.5}},
        "versions": {"wikitext": 2},
    }

    evaluate._write_results(
        config=config,
        model_id="rwkv-current",
        version="0.4.12",
        results=raw,
    )

    assert json.loads((output_dir / "results.json").read_text()) == raw
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["metrics"] == raw["results"]
    assert summary["model_id"] == "rwkv-current"
    assert summary["effective_max_length"] == 10238
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((output_dir / "results.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output_dir / "summary.json").stat().st_mode) == 0o600
