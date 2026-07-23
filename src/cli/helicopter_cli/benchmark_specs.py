from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


EXPECTED_FIELDS = ("knowledge", "math", "coding", "instruction_following")
EXPECTED_PER_FIELD = 30
ALLOWED_FORMATS = {
    "choice",
    "open_qa",
    "math_boxed",
    "math_choice",
    "formal_proof",
    "diff_patch",
    "python_program",
    "python_function",
    "python_snippet",
    "generic_code",
    "code_completion",
    "code_reasoning",
    "code_retrieval",
    "instruction",
}
SAMPLING_FIELDS = {
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "penalty_decay",
    "stop",
}
EVALUATION_FIELDS = {
    "metric",
    "avg_k",
    "rollout_n",
    "target_generations_per_benchmark",
    "large_benchmark_generation_threshold",
    "large_benchmark_sample_rate",
}


def _table(value: Any, *, key: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: [{key}] must be a TOML table")
    return value


def _contains_checker(value: Any) -> bool:
    if isinstance(value, dict):
        return any("checker" in str(key).casefold() or _contains_checker(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_checker(child) for child in value)
    return False


def load_benchmark_index(index_path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate a one-file-per-benchmark index."""

    index_path = index_path.resolve()
    with index_path.open("rb") as file:
        index = tomllib.load(file)
    files = index.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(item, str) for item in files):
        raise ValueError(f"{index_path}: files must be a non-empty TOML string array")
    expected_per_field = int(index.get("target_per_domain", EXPECTED_PER_FIELD))
    if index.get("scoring") != "judge":
        raise ValueError(f"{index_path}: top-level scoring must be 'judge'")

    specs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    seen_tasks: set[str] = set()
    ordinals: dict[str, set[int]] = {field: set() for field in EXPECTED_FIELDS}
    for relative in files:
        path = (index_path.parent / relative).resolve()
        if path in seen_paths:
            raise ValueError(f"{index_path}: duplicate benchmark file {relative}")
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"{index_path}: benchmark file not found: {path}")
        with path.open("rb") as file:
            spec = tomllib.load(file)
        if _contains_checker(spec):
            raise ValueError(f"{path}: checker keys are forbidden; final scoring is Judge-only")

        benchmark = _table(spec.get("benchmark"), key="benchmark", path=path)
        dataset = _table(spec.get("dataset"), key="dataset", path=path)
        prompt = _table(spec.get("prompt"), key="prompt", path=path)
        judge = _table(spec.get("judge"), key="judge", path=path)
        evaluation = _table(spec.get("evaluation"), key="evaluation", path=path)
        sampling = _table(spec.get("sampling"), key="sampling", path=path)

        field = str(benchmark.get("field", ""))
        if field not in EXPECTED_FIELDS:
            raise ValueError(f"{path}: unknown benchmark field {field!r}")
        ordinal = int(benchmark.get("ordinal", 0))
        name = str(benchmark.get("name", "")).strip()
        task = str(benchmark.get("task", "")).strip()
        if not name or name in seen_names:
            raise ValueError(f"{path}: benchmark.name must be non-empty and unique")
        if not task or task in seen_tasks:
            raise ValueError(f"{path}: benchmark.task must be non-empty and unique")
        if ordinal in ordinals[field]:
            raise ValueError(f"{path}: duplicate ordinal {ordinal} in field {field}")
        seen_names.add(name)
        seen_tasks.add(task)
        ordinals[field].add(ordinal)

        modes = benchmark.get("allowed_modes")
        if not isinstance(modes, list) or not modes or not all(isinstance(item, str) for item in modes):
            raise ValueError(f"{path}: benchmark.allowed_modes must be a non-empty string array")
        request_format = str(prompt.get("format", ""))
        if request_format not in ALLOWED_FORMATS:
            raise ValueError(f"{path}: unsupported prompt.format {request_format!r}")
        if prompt.get("raw_question_only") is not True or prompt.get("add_instructions") is not False:
            raise ValueError(f"{path}: prompt must preserve the raw official question without added instructions")
        if dataset.get("prompt_contract") != "raw_question_only" or dataset.get("gold_contract") != "final_answer_only":
            raise ValueError(f"{path}: dataset prompt/gold contract must be raw question + final answer")
        if not str(dataset.get("source", "")).strip():
            raise ValueError(f"{path}: dataset.source is required")
        if judge.get("enabled") is not True or judge.get("contract") != "reference_candidate":
            raise ValueError(f"{path}: Judge must use the reference_candidate contract")
        if not isinstance(judge.get("primary_score"), bool):
            raise ValueError(f"{path}: judge.primary_score must be a TOML boolean")
        if evaluation.get("metric") != "avg":
            raise ValueError(f"{path}: evaluation.metric must be 'avg'")
        avg_k = int(evaluation.get("avg_k", 0))
        rollout_n = int(evaluation.get("rollout_n", 0))
        if avg_k <= 0 or rollout_n != avg_k:
            raise ValueError(f"{path}: evaluation requires positive rollout_n == avg_k")
        unknown_evaluation = set(evaluation) - EVALUATION_FIELDS
        if unknown_evaluation:
            raise ValueError(f"{path}: unsupported evaluation fields: {sorted(unknown_evaluation)}")
        unknown_sampling = set(sampling) - SAMPLING_FIELDS
        if unknown_sampling:
            raise ValueError(f"{path}: unsupported sampling fields: {sorted(unknown_sampling)}")

        loaded = dict(spec)
        loaded["_path"] = str(path)
        specs.append(loaded)

    counts = Counter(str(spec["benchmark"]["field"]) for spec in specs)
    expected_counts = {field: expected_per_field for field in EXPECTED_FIELDS}
    if dict(counts) != expected_counts:
        raise ValueError(f"{index_path}: expected {expected_counts}, got {dict(counts)}")
    for field in EXPECTED_FIELDS:
        expected_ordinals = set(range(1, expected_per_field + 1))
        if ordinals[field] != expected_ordinals:
            raise ValueError(f"{index_path}: {field} ordinals must be 1..{expected_per_field}")
    return specs


def benchmark_specs_by_task(specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(spec["benchmark"]["task"]): spec for spec in specs}


__all__ = [
    "ALLOWED_FORMATS",
    "EXPECTED_FIELDS",
    "EXPECTED_PER_FIELD",
    "benchmark_specs_by_task",
    "load_benchmark_index",
]
