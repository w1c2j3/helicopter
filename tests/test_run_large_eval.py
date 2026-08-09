from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_large_eval", ROOT / "scripts/run_large_eval.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_model_queue_expands_modes_without_cross_model_barrier() -> None:
    manifest = tomllib.loads((ROOT / "configs/large_eval_60.toml").read_text())
    run = manifest["run"]
    benchmarks = manifest["benchmarks"]
    jobs = MODULE.build_job_queue(
        benchmarks,
        base_modes=run["base_modes"],
        cot_fields=set(run["cot_fields"]),
        cot_modes=run["cot_modes"],
    )

    assert len(benchmarks) == 60
    active_benchmarks = [entry for entry in benchmarks if not entry.get("skip", False)]
    expected_jobs = sum(
        4 if entry["field"] in set(run["cot_fields"]) else 2
        for entry in active_benchmarks
    )
    assert len(jobs) == expected_jobs

    first_active_math = next(
        entry["task"] for entry in active_benchmarks if entry["field"] == "math"
    )
    first_task = [mode for entry, mode in jobs if entry["task"] == first_active_math]
    assert first_task == ["naive_nocot", "normal_nocot", "naive_cot", "normal_cot"]

    coding_modes = {
        mode for entry, mode in jobs if entry["field"] == "coding"
    }
    instruction_modes = {
        mode for entry, mode in jobs if entry["field"] == "instruction_following"
    }
    assert coding_modes == {"naive_nocot", "normal_nocot"}
    assert instruction_modes == {"naive_nocot", "normal_nocot"}
