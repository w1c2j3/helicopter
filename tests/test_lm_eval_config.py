from __future__ import annotations

import json
from pathlib import Path

import pytest

from helicopter_lm_eval.config import ConfigError, LMEvalConfig


def _manifest(tmp_path: Path, max_model_len: int = 16) -> Path:
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "global_step": 9,
                "wkv_mode": "fp32io16",
                "vllm_version": "0.23.1.dev0",
                "max_model_len": max_model_len,
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


def _config(
    tmp_path: Path,
    *,
    batch_size: int = 2,
    eot_token_id: int = 0,
    max_gen_toks: int = 8,
    extra: str = "",
) -> Path:
    path = tmp_path / "eval.toml"
    path.write_text(
        "\n".join(
            [
                "schema_version = 1",
                'backend = "vllm_http"',
                'tasks = ["wikitext"]',
                f'output_dir = "{tmp_path / "results"}"',
                f"batch_size = {batch_size}",
                f"eot_token_id = {eot_token_id}",
                f"max_gen_toks = {max_gen_toks}",
                "log_samples = false",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_config_loads_ppl_settings_and_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = LMEvalConfig.read(
        _config(tmp_path),
        {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
    )

    assert config.tasks == ("wikitext",)
    assert config.batch_size == 2
    assert config.eot_token_id == 0
    assert config.max_gen_toks == 8
    assert config.limit is None
    assert config.manifest.global_step == 9


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": "unknown = true"}, "unknown lm-eval config fields"),
        ({"eot_token_id": -1}, "eot_token_id must be a non-negative integer"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"max_gen_toks": 0}, "max_gen_toks must be a positive integer"),
        ({"extra": "limit = 0"}, "limit must be a positive integer"),
    ],
)
def test_config_rejects_invalid_fields(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ConfigError, match=message):
        LMEvalConfig.read(
            _config(tmp_path, **overrides),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
        )


def test_config_requires_a_regular_manifest(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a regular file"):
        LMEvalConfig.read(
            _config(tmp_path),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(tmp_path / "missing.json")},
        )


def test_config_rejects_output_path_that_is_not_a_directory(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "results").write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConfigError, match="output_dir must be a regular directory"):
        LMEvalConfig.read(
            _config(tmp_path),
            {"HELICOPTER_VLLM_POOL_MANIFEST": str(manifest)},
        )
