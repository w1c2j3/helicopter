from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from helicopter_cli import commands, config


ROOT = Path(__file__).resolve().parents[1]


def test_mode_specific_max_tokens_are_resolved_before_model_args() -> None:
    config_path = "configs/benchmarks/g1h/math/051_math_algebra.toml"
    loaded, _ = config.load_config(ROOT, config_path)
    expected = {
        "naive_nocot": 8192,
        "normal_nocot": 8192,
        "naive_cot": 8192,
        "normal_cot": 8192,
    }
    for mode, max_tokens in expected.items():
        args = SimpleNamespace(
            prompt_mode=mode,
            max_tokens=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            seed=None,
            repetition_penalty=None,
            frequency_penalty=None,
            presence_penalty=None,
            penalty_decay=None,
            stop=None,
        )
        sampling = commands.resolve_lighteval_sampling(args, env={}, config=loaded)
        assert sampling["max_tokens"] == max_tokens
