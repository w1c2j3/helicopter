from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


EXPECTED_FIELDS = ("knowledge", "math", "coding", "instruction_following")
EXPECTED_PER_FIELD = 100
ALLOWED_FORMATS = {
    "choice",
    "open_qa",
    "pubmedqa",
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
    "multiturn",
}
SAMPLING_FIELDS = {
    "max_tokens",
    "context_budget",
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
PROMPT_MODES = {
    "naive_cot",
    "normal_cot",
    "naive_nocot",
    "normal_nocot",
}
EVALUATION_FIELDS = {
    "metric",
    "avg_k",
    "rollout_n",
    "pass_k",
    "pass_n",
    "gpass_k",
    "gpass_n",
    "native_n",
    "generation_size",
    "gpass_generation_size",
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


def load_benchmark_spec(spec_path: Path) -> dict[str, Any]:
    """Load and validate one self-contained benchmark TOML file."""

    spec_path = spec_path.resolve()
    if not spec_path.is_file():
        raise ValueError(f"benchmark file not found: {spec_path}")
    with spec_path.open("rb") as file:
        spec = tomllib.load(file)
    if _contains_checker(spec):
        raise ValueError(
            f"{spec_path}: checker keys are forbidden; scoring must come from the "
            "LightEval task's native metric"
        )

    benchmark = _table(spec.get("benchmark"), key="benchmark", path=spec_path)
    dataset = _table(spec.get("dataset"), key="dataset", path=spec_path)
    prompt = _table(spec.get("prompt"), key="prompt", path=spec_path)
    scoring = _table(spec.get("scoring"), key="scoring", path=spec_path)
    evaluation = _table(spec.get("evaluation"), key="evaluation", path=spec_path)
    sampling = _table(spec.get("sampling"), key="sampling", path=spec_path)

    field = str(benchmark.get("field", ""))
    if field not in EXPECTED_FIELDS:
        raise ValueError(f"{spec_path}: unknown benchmark field {field!r}")
    ordinal = int(benchmark.get("ordinal", 0))
    name = str(benchmark.get("name", "")).strip()
    task = str(benchmark.get("task", "")).strip()
    if ordinal <= 0:
        raise ValueError(f"{spec_path}: benchmark.ordinal must be positive")
    if not name:
        raise ValueError(f"{spec_path}: benchmark.name must be non-empty")
    if not task:
        raise ValueError(f"{spec_path}: benchmark.task must be non-empty")

    modes = benchmark.get("allowed_modes")
    if not isinstance(modes, list) or not modes or not all(isinstance(item, str) for item in modes):
        raise ValueError(f"{spec_path}: benchmark.allowed_modes must be a non-empty string array")
    unknown_modes = set(modes) - PROMPT_MODES
    if unknown_modes:
        raise ValueError(f"{spec_path}: unsupported benchmark.allowed_modes: {sorted(unknown_modes)}")
    request_format = str(prompt.get("format", ""))
    if request_format not in ALLOWED_FORMATS:
        raise ValueError(f"{spec_path}: unsupported prompt.format {request_format!r}")
    gold_contract = str(dataset.get("gold_contract", ""))
    allowed_gold_contracts = (
        {"final_answer_only", "constraint_list"}
        if field == "instruction_following"
        else {"final_answer_only"}
    )
    if gold_contract not in allowed_gold_contracts:
        raise ValueError(
            f"{spec_path}: dataset.gold_contract must be one of {sorted(allowed_gold_contracts)}"
        )
    if not str(dataset.get("source", "")).strip():
        raise ValueError(f"{spec_path}: dataset.source is required")
    if scoring != {"provider": "lighteval", "metric_source": "task_default"}:
        raise ValueError(f"{spec_path}: scoring must use the LightEval task's default metric")

    evaluation_metric = str(evaluation.get("metric", "")).strip()
    if evaluation_metric not in {"avg", "native"}:
        raise ValueError(f"{spec_path}: evaluation.metric must be 'avg' or 'native'")
    avg_k = int(evaluation.get("avg_k", 0))
    rollout_n = int(evaluation.get("rollout_n", 0))
    if avg_k <= 0 or rollout_n <= 0:
        raise ValueError(f"{spec_path}: evaluation requires positive avg_k and rollout_n")
    if evaluation_metric == "avg" and rollout_n != avg_k:
        raise ValueError(f"{spec_path}: avg evaluation requires rollout_n == avg_k")
    positive_evaluation_fields = (
        "avg_k",
        "rollout_n",
        "pass_k",
        "pass_n",
        "gpass_k",
        "gpass_n",
        "native_n",
        "generation_size",
        "gpass_generation_size",
        "target_generations_per_benchmark",
        "large_benchmark_generation_threshold",
    )
    for key in positive_evaluation_fields:
        if key not in evaluation:
            continue
        value = evaluation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{spec_path}: evaluation.{key} must be a positive integer")
    if "pass_k" in evaluation and "pass_n" in evaluation and int(evaluation["pass_k"]) > int(evaluation["pass_n"]):
        raise ValueError(f"{spec_path}: evaluation.pass_k cannot exceed pass_n")
    if "gpass_k" in evaluation and "gpass_n" in evaluation and int(evaluation["gpass_k"]) > int(evaluation["gpass_n"]):
        raise ValueError(f"{spec_path}: evaluation.gpass_k cannot exceed gpass_n")
    unknown_evaluation = set(evaluation) - EVALUATION_FIELDS
    if unknown_evaluation:
        raise ValueError(f"{spec_path}: unsupported evaluation fields: {sorted(unknown_evaluation)}")
    unknown_sampling = set(sampling) - SAMPLING_FIELDS
    if unknown_sampling:
        raise ValueError(f"{spec_path}: unsupported sampling fields: {sorted(unknown_sampling)}")

    max_tokens = sampling.get("max_tokens")
    if isinstance(max_tokens, dict):
        if set(max_tokens) != set(modes):
            raise ValueError(
                f"{spec_path}: sampling.max_tokens mode keys must exactly match benchmark.allowed_modes"
            )
        invalid_budgets = {
            mode: value
            for mode, value in max_tokens.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        }
        if invalid_budgets:
            raise ValueError(
                f"{spec_path}: sampling.max_tokens mode budgets must be positive integers: {invalid_budgets}"
            )
    elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"{spec_path}: sampling.max_tokens must be a positive integer or mode table")
    context_budget = sampling.get("context_budget")
    if context_budget is not None and (
        isinstance(context_budget, bool) or not isinstance(context_budget, int) or context_budget <= 0
    ):
        raise ValueError(f"{spec_path}: sampling.context_budget must be a positive integer")

    loaded = dict(spec)
    loaded["_path"] = str(spec_path)
    return loaded


def load_benchmark_index(index_path: Path) -> list[dict[str, Any]]:
    """Load and strictly validate a one-file-per-benchmark index."""

    index_path = index_path.resolve()
    with index_path.open("rb") as file:
        index = tomllib.load(file)
    files = index.get("files")
    if not isinstance(files, list) or not files or not all(isinstance(item, str) for item in files):
        raise ValueError(f"{index_path}: files must be a non-empty TOML string array")
    expected_per_field = int(index.get("target_per_domain", EXPECTED_PER_FIELD))
    if index.get("scoring") != "lighteval":
        raise ValueError(f"{index_path}: top-level scoring must be 'lighteval'")

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
        spec = load_benchmark_spec(path)
        benchmark = spec["benchmark"]
        field = str(benchmark["field"])
        ordinal = int(benchmark["ordinal"])
        name = str(benchmark["name"])
        task = str(benchmark["task"])
        if name in seen_names:
            raise ValueError(f"{path}: benchmark.name must be unique")
        if task in seen_tasks:
            raise ValueError(f"{path}: benchmark.task must be unique")
        if ordinal in ordinals[field]:
            raise ValueError(f"{path}: duplicate ordinal {ordinal} in field {field}")
        seen_names.add(name)
        seen_tasks.add(task)
        ordinals[field].add(ordinal)
        specs.append(spec)

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
    "load_benchmark_spec",
]
