"""Publish already-completed lm-eval result directories.

This module deliberately uses a separate campaign contract from live evaluation.
Existing artifacts may contain only one WKV mode, but they still go through the
same authenticated task publication, canonical idempotency, and finalize path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
import uuid

from helicopter_lighteval.publish import (
    PublicationError,
    ScoreboardClient,
    content_digest,
)

from .config import ConfigError, PromptConfig
from .publish import (
    LM_EVAL_VERSION,
    _aggregates,
    _detail,
    _details_path,
    _diagnostics,
    _document_counts,
)


EXISTING_CAMPAIGN_SCHEMA = "lm-eval-existing-campaign-v1"
_WKV_MODES = frozenset({"fp16", "fp32io16"})
_PROMPT_PROFILES = frozenset({"bot", "assistant", "function_calling", "none"})


class ExistingPublicationError(PublicationError):
    """Raised when an existing artifact set cannot satisfy the import contract."""


@dataclass(frozen=True)
class ExistingUnit:
    output_dir: Path
    results: dict[str, object]
    summary: dict[str, object]
    artifact: dict[str, object]
    results_sha256: str
    summary_sha256: str
    artifact_sha256: str
    weight_sha256: str
    weight_display_name: str
    wkv_mode: str
    source_timestamp: str | float | None
    vllm_version: str
    torch_version: str


def _read_hashed_json(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExistingPublicationError(f"invalid lm-eval artifact: {path}") from error
    if not isinstance(value, dict):
        raise ExistingPublicationError(f"lm-eval artifact must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _strings(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return sorted(
            {
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            }
        )
    return []


def _positive_int(value: object, *, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExistingPublicationError(f"{name} must be a positive integer")
    return value


def _source_module(task_name: str, task_config: Mapping[str, object]) -> str:
    metadata = task_config.get("metadata")
    source = metadata.get("config_source") if isinstance(metadata, Mapping) else None
    if not isinstance(source, str) or not source.strip():
        return f"historical/{task_name}.yaml"
    normalized = source.replace("\\", "/")
    marker = "/tasks/"
    if marker in normalized:
        return "tasks/" + normalized.split(marker, 1)[1]
    return Path(normalized).name


def _module_family(task_name: str, module: str) -> str:
    parent = Path(module).parent.name
    return task_name if parent in {"", ".", "tasks"} else parent


def _selector_for(
    task_name: str,
    summary: Mapping[str, object],
    results: Mapping[str, object],
) -> str:
    raw_roots = summary.get("tasks")
    roots = (
        [value for value in raw_roots if isinstance(value, str)]
        if isinstance(raw_roots, list)
        else []
    )
    if task_name in roots:
        return task_name
    groups = results.get("group_subtasks")
    if isinstance(groups, Mapping):
        for root in roots:
            members = groups.get(root)
            if isinstance(members, list) and task_name in members:
                return root
    if len(roots) == 1:
        return roots[0]
    return task_name


def _benchmark_config(
    summary: Mapping[str, object], selector: str
) -> dict[str, object]:
    values = summary.get("benchmark_configs")
    if isinstance(values, list):
        for value in values:
            if isinstance(value, dict) and value.get("selector") == selector:
                return dict(value)
    return {}


def _prompt(summary: Mapping[str, object], selector: str) -> dict[str, object]:
    benchmark = _benchmark_config(summary, selector)
    value = benchmark.get("prompt")
    if not isinstance(value, dict):
        value = summary.get("prompt")
    if not isinstance(value, dict):
        return PromptConfig().public()
    profile = value.get("profile", "none")
    if profile not in _PROMPT_PROFILES:
        raise ExistingPublicationError(
            f"unsupported prompt profile in existing artifact: {profile!r}"
        )
    prompt = dict(value)
    prompt["profile"] = profile
    return prompt


def _expected_task(
    *,
    unit: ExistingUnit,
    task_name: str,
    selector: str,
    task_config: Mapping[str, object],
    results: Mapping[str, object],
) -> dict[str, object]:
    metadata = task_config.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    versions = results.get("versions")
    version = versions.get(task_name) if isinstance(versions, Mapping) else None
    if version is None:
        version = metadata.get("version", "0")
    module = _source_module(task_name, task_config)
    splits = [
        str(task_config[name])
        for name in ("test_split", "validation_split")
        if isinstance(task_config.get(name), str) and task_config[name].strip()
    ]
    return {
        "identity": f"{unit.weight_sha256}:{unit.wkv_mode}:{task_name}",
        "weight_sha256": unit.weight_sha256,
        "weight_display_name": unit.weight_display_name,
        "wkv_mode": unit.wkv_mode,
        "selector": selector,
        "task_name": task_name,
        "task_version": str(version),
        "module_family": _module_family(task_name, module),
        "module": module,
        "dataset": str(task_config.get("dataset_path") or "unknown"),
        "subset": str(task_config.get("dataset_name") or ""),
        "evaluation_splits": list(dict.fromkeys(splits or ["unknown"])),
        "languages": _strings(metadata.get("languages")),
        "upstream_tags": _strings(metadata.get("tags")),
    }


def _validate_artifact_shape(
    *,
    output_dir: Path,
    results: Mapping[str, object],
    artifact: Mapping[str, object],
) -> None:
    result_map = results.get("results")
    sample_map = results.get("samples")
    config_map = results.get("configs")
    counts = results.get("n-samples")
    if not isinstance(result_map, Mapping):
        raise ExistingPublicationError(f"results.json lacks results: {output_dir}")
    if not isinstance(sample_map, Mapping):
        raise ExistingPublicationError(f"results.json lacks samples: {output_dir}")
    if not isinstance(config_map, Mapping):
        raise ExistingPublicationError(f"results.json lacks configs: {output_dir}")
    if not isinstance(counts, Mapping):
        raise ExistingPublicationError(f"results.json lacks n-samples: {output_dir}")
    task_names = set(sample_map)
    if not task_names or not all(isinstance(name, str) for name in task_names):
        raise ExistingPublicationError(
            f"results.json has no valid sample tasks: {output_dir}"
        )
    missing = sorted((task_names - set(result_map)) | (task_names - set(config_map)))
    if missing:
        raise ExistingPublicationError(
            f"results.json is missing result/config entries for {', '.join(missing)}"
        )
    for task_name in sorted(task_names):
        rows = sample_map[task_name]
        if not isinstance(rows, list) or not rows:
            raise ExistingPublicationError(f"task has no samples: {task_name}")
        if not isinstance(config_map[task_name], Mapping):
            raise ExistingPublicationError(f"task config is not an object: {task_name}")
        if not isinstance(counts.get(task_name), Mapping):
            raise ExistingPublicationError(
                f"task sample counts are missing: {task_name}"
            )
        try:
            details = [_detail(task_name, index, row) for index, row in enumerate(rows)]
            _document_counts(counts, task_name, details)
            details_path = _details_path(artifact, task_name)
            _validate_relative_file(output_dir, details_path, task_name=task_name)
            aggregates = _aggregates(result_map[task_name])
        except PublicationError as error:
            raise ExistingPublicationError(
                f"invalid existing task {task_name} in {output_dir}: {error}"
            ) from error
        if not aggregates:
            raise ExistingPublicationError(f"task has no finite metrics: {task_name}")


def _resolve_weight(
    summary: Mapping[str, object],
    *,
    output_dir: Path,
    weight_sha256: str | None,
    weight_display_name: str | None,
) -> tuple[str, str]:
    raw_hash = summary.get("weight_sha256")
    resolved_hash = weight_sha256 or (raw_hash if isinstance(raw_hash, str) else None)
    if (
        not isinstance(resolved_hash, str)
        or len(resolved_hash) != 64
        or any(character not in "0123456789abcdef" for character in resolved_hash)
    ):
        raise ExistingPublicationError(
            f"{output_dir}/summary.json lacks a valid weight_sha256; "
            "pass --weight-sha256 explicitly"
        )
    if isinstance(raw_hash, str) and raw_hash and raw_hash != resolved_hash:
        raise ExistingPublicationError(
            f"weight SHA override conflicts with summary.json: {output_dir}"
        )
    raw_name = summary.get("weight_display_name")
    resolved_name = weight_display_name or (
        raw_name if isinstance(raw_name, str) and raw_name.strip() else None
    )
    if resolved_name is None:
        model_id = summary.get("model_id")
        resolved_name = (
            str(model_id).strip()
            if isinstance(model_id, str)
            else "existing-artifact"
        )
    if (
        not resolved_name
        or resolved_name != resolved_name.strip()
        or len(resolved_name) > 500
    ):
        raise ExistingPublicationError(
            f"weight display name must be a trimmed string up to 500 characters: "
            f"{output_dir}"
        )
    return resolved_hash, resolved_name


def _source_timestamp(
    summary: Mapping[str, object], results: Mapping[str, object]
) -> str | float | None:
    for value in (summary.get("date"), results.get("date")):
        if isinstance(value, str) and value and value == value.strip():
            return value
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
        ):
            try:
                timestamp = float(value)
            except OverflowError:
                continue
            if math.isfinite(timestamp):
                return timestamp
    return None


def _validate_relative_file(
    output_dir: Path,
    raw_path: object,
    *,
    task_name: str,
) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise ExistingPublicationError(
            f"artifact path for {task_name} must be a non-empty string"
        )
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != raw_path
    ):
        raise ExistingPublicationError(
            f"artifact path for {task_name} must be a normalized relative path"
        )
    candidate = output_dir.joinpath(*path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ExistingPublicationError(
            f"artifact path for {task_name} does not exist: {raw_path}"
        ) from error
    if candidate.is_symlink() or not resolved.is_file():
        raise ExistingPublicationError(
            f"artifact path for {task_name} is not a regular file: {raw_path}"
        )
    try:
        resolved.relative_to(output_dir)
    except ValueError as error:
        raise ExistingPublicationError(
            f"artifact path for {task_name} escapes output directory: {raw_path}"
        ) from error
    return raw_path


def load_existing_unit(
    output_dir: Path,
    *,
    weight_sha256: str | None = None,
    weight_display_name: str | None = None,
    vllm_version: str = "not-recorded-in-artifact",
    torch_version: str = "not-recorded-in-artifact",
) -> ExistingUnit:
    for name, version in (("vllm", vllm_version), ("torch", torch_version)):
        if not version or version != version.strip():
            raise ExistingPublicationError(
                f"{name} version must be a non-empty trimmed string"
            )
    expanded_output_dir = output_dir.expanduser()
    if expanded_output_dir.is_symlink():
        raise ExistingPublicationError(
            f"existing output directory must not be a symlink: {expanded_output_dir}"
        )
    output_dir = expanded_output_dir.resolve()
    if not output_dir.is_dir():
        raise ExistingPublicationError(
            f"existing output directory is invalid: {output_dir}"
        )
    paths = {
        name: output_dir / name
        for name in ("results.json", "summary.json", "artifacts.json")
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ExistingPublicationError(
                f"existing artifact is missing {name}: {output_dir}"
            )
    results, results_sha256 = _read_hashed_json(paths["results.json"])
    summary, summary_sha256 = _read_hashed_json(paths["summary.json"])
    artifact, artifact_sha256 = _read_hashed_json(paths["artifacts.json"])
    if artifact.get("schema_version") not in {1, 2}:
        raise ExistingPublicationError(
            f"artifacts.json has unsupported schema_version: {output_dir}"
        )
    evaluator = artifact.get("evaluator")
    if not isinstance(evaluator, Mapping) or (
        evaluator.get("name") != "lm-eval"
        or evaluator.get("version") != LM_EVAL_VERSION
    ):
        raise ExistingPublicationError(
            f"artifact evaluator must be lm-eval {LM_EVAL_VERSION}: {output_dir}"
        )
    if (
        artifact.get("results_path") != "results.json"
        or artifact.get("summary_path") != "summary.json"
    ):
        raise ExistingPublicationError(
            f"artifacts.json must point to results.json and summary.json: {output_dir}"
        )
    mode = summary.get("wkv_mode")
    if mode not in _WKV_MODES:
        raise ExistingPublicationError(
            f"summary.json has invalid wkv_mode: {output_dir}"
        )
    resolved_hash, resolved_name = _resolve_weight(
        summary,
        output_dir=output_dir,
        weight_sha256=weight_sha256,
        weight_display_name=weight_display_name,
    )
    _validate_artifact_shape(output_dir=output_dir, results=results, artifact=artifact)
    return ExistingUnit(
        output_dir=output_dir,
        results=results,
        summary=summary,
        artifact=artifact,
        results_sha256=results_sha256,
        summary_sha256=summary_sha256,
        artifact_sha256=artifact_sha256,
        weight_sha256=resolved_hash,
        weight_display_name=resolved_name,
        wkv_mode=str(mode),
        source_timestamp=_source_timestamp(summary, results),
        vllm_version=vllm_version,
        torch_version=torch_version,
    )


def _unit_tasks(unit: ExistingUnit) -> list[dict[str, object]]:
    sample_map = unit.results["samples"]
    config_map = unit.results["configs"]
    assert isinstance(sample_map, Mapping)
    assert isinstance(config_map, Mapping)
    return [
        _expected_task(
            unit=unit,
            task_name=task_name,
            selector=_selector_for(task_name, unit.summary, unit.results),
            task_config=config_map[task_name],
            results=unit.results,
        )
        for task_name in sorted(sample_map)
    ]


def _metadata_key(task: Mapping[str, object]) -> tuple[object, ...]:
    return (
        task["task_name"],
        task["selector"],
        task["task_version"],
        task["module_family"],
        task["module"],
        task["dataset"],
        task["subset"],
        tuple(task["evaluation_splits"]),
        tuple(task["languages"]),
        tuple(task["upstream_tags"]),
    )


def _validate_matrix(
    units: Sequence[ExistingUnit],
    tasks_by_unit: Mapping[Path, list[dict[str, object]]],
) -> list[dict[str, object]]:
    if not units:
        raise ExistingPublicationError(
            "at least one existing output directory is required"
        )
    identities: set[str] = set()
    metadata_by_name: dict[str, tuple[object, ...]] = {}
    task_names_by_unit: list[set[str]] = []
    for unit in units:
        tasks = tasks_by_unit[unit.output_dir]
        names: set[str] = set()
        for task in tasks:
            identity = str(task["identity"])
            if identity in identities:
                raise ExistingPublicationError(f"duplicate task identity: {identity}")
            identities.add(identity)
            task_name = str(task["task_name"])
            names.add(task_name)
            key = _metadata_key(task)
            previous = metadata_by_name.setdefault(task_name, key)
            if previous != key:
                raise ExistingPublicationError(
                    "task metadata differs between existing output directories: "
                    f"{task_name}"
                )
        task_names_by_unit.append(names)
    if len({tuple(sorted(names)) for names in task_names_by_unit}) != 1:
        raise ExistingPublicationError(
            "existing output directories must contain the same task set"
        )
    return sorted(
        [task for unit in units for task in tasks_by_unit[unit.output_dir]],
        key=lambda task: str(task["identity"]),
    )


def _campaign_payload(
    *,
    units: Sequence[ExistingUnit],
    expected_tasks: list[dict[str, object]],
) -> dict[str, object]:
    sources = [
        {
            "results_sha256": unit.results_sha256,
            "summary_sha256": unit.summary_sha256,
            "artifacts_sha256": unit.artifact_sha256,
            "weight_sha256": unit.weight_sha256,
            "weight_display_name": unit.weight_display_name,
            "wkv_mode": unit.wkv_mode,
            "source_timestamp": unit.source_timestamp,
            "vllm_version": unit.vllm_version,
            "torch_version": unit.torch_version,
        }
        for unit in sorted(
            units,
            key=lambda value: (
                value.weight_sha256,
                value.wkv_mode,
                value.results_sha256,
            ),
        )
    ]
    selectors = sorted({str(task["selector"]) for task in expected_tasks})
    run_key = content_digest(
        {
            "schema_version": EXISTING_CAMPAIGN_SCHEMA,
            "sources": sources,
        }
    )
    return {
        "schema_version": EXISTING_CAMPAIGN_SCHEMA,
        "run_key": run_key,
        "config_digest": content_digest({"sources": sources, "selectors": selectors}),
        "registry_digest": content_digest(expected_tasks),
        "eval_contract_digest": content_digest(
            {
                "schema_version": EXISTING_CAMPAIGN_SCHEMA,
                "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
                "source_artifacts_immutable": True,
            }
        ),
        "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
        "configured_selectors": selectors,
        "resolved_selectors": selectors,
        "skipped_selectors": [],
        "expected_tasks": expected_tasks,
    }


def _task_payload(
    *,
    unit: ExistingUnit,
    task: Mapping[str, object],
    campaign_id: str,
) -> dict[str, object]:
    task_name = str(task["task_name"])
    results = unit.results
    summary = unit.summary
    artifact = unit.artifact
    samples = results["samples"]
    configs = results["configs"]
    counts = results["n-samples"]
    result_map = results["results"]
    assert isinstance(samples, Mapping)
    assert isinstance(configs, Mapping)
    assert isinstance(counts, Mapping)
    assert isinstance(result_map, Mapping)
    rows = samples[task_name]
    config = configs[task_name]
    if not isinstance(rows, list) or not isinstance(config, Mapping):
        raise ExistingPublicationError(f"existing task is malformed: {task_name}")
    details = [_detail(task_name, index, row) for index, row in enumerate(rows)]
    original_docs, effective_docs = _document_counts(counts, task_name, details)
    task_config = dict(config)
    task_config.update(
        original_num_docs=original_docs,
        effective_num_docs=effective_docs,
        skipped_multiselect_docs=original_docs - effective_docs,
    )
    aggregates = _aggregates(result_map[task_name])
    primary_metric = next((name for name in aggregates if "stderr" not in name), None)
    if primary_metric is None:
        raise ExistingPublicationError(
            f"existing task has no primary metric: {task_name}"
        )
    selector = str(task["selector"])
    benchmark = _benchmark_config(summary, selector)
    prompt = _prompt(summary, selector)
    root_config = results.get("config")
    root_config = root_config if isinstance(root_config, Mapping) else {}
    batch_size = _positive_int(
        benchmark.get(
            "batch_size",
            task_config.get("batch_size", root_config.get("batch_size")),
        ),
        name=f"batch_size for {task_name}",
        default=1,
    )
    max_gen_toks = _positive_int(
        benchmark.get("max_gen_toks", task_config.get("max_gen_toks")),
        name=f"max_gen_toks for {task_name}",
        default=256,
    )
    max_model_len = _positive_int(
        summary.get("max_model_len"), name=f"max_model_len for {unit.output_dir}"
    )
    eot_token_id = summary.get("eot_token_id")
    if eot_token_id is None:
        eot_token_id = 0
    if (
        isinstance(eot_token_id, bool)
        or not isinstance(eot_token_id, int)
        or eot_token_id < 0
    ):
        raise ExistingPublicationError(f"eot_token_id is invalid: {unit.output_dir}")
    generation_kwargs = task_config.get("generation_kwargs", {})
    if not isinstance(generation_kwargs, Mapping):
        generation_kwargs = {}
    generation_override = benchmark.get(
        "generation_kwargs", summary.get("generation_kwargs", {})
    )
    if not isinstance(generation_override, Mapping):
        generation_override = {}
    return {
        "schema_version": "lm-eval-task-v1",
        "campaign_id": campaign_id,
        "task": dict(task),
        "artifact": {
            "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
            "results_path": "results.json",
            "details_paths": [_details_path(artifact, task_name)],
        },
        "task_config": task_config,
        "model": {
            "weight_sha256": unit.weight_sha256,
            "weight_display_name": unit.weight_display_name,
            "wkv_mode": unit.wkv_mode,
            "prompt_template": prompt["profile"],
            "gemm_policy": "fp16-accumulation"
            if unit.wkv_mode == "fp16"
            else "fp32-accumulation",
            "gpu": "existing-evaluation-artifact",
            "max_num_seqs": batch_size,
            "max_num_batched_tokens": max_model_len,
            "dependency_versions": {
                "lm-eval": LM_EVAL_VERSION,
                "vllm": unit.vllm_version,
                "torch": unit.torch_version,
            },
            "evaluator": "lm-eval",
        },
        "sampling_config": {
            "output_type": task_config.get("output_type"),
            "batch_size": batch_size,
            "eot_token_id": eot_token_id,
            "default_max_gen_toks": max_gen_toks,
            "generation_kwargs": dict(generation_kwargs),
            "generation_kwargs_override": dict(generation_override),
            "prompt": prompt,
            "existing_artifact": True,
            "source_artifacts": {
                "results_sha256": unit.results_sha256,
                "summary_sha256": unit.summary_sha256,
                "artifacts_sha256": unit.artifact_sha256,
                "timestamp": unit.source_timestamp,
            },
        },
        "primary_metric": primary_metric,
        "aggregates": aggregates,
        "diagnostics": _diagnostics(details),
        "details": details,
    }


def publish_existing(
    *,
    output_dirs: Sequence[Path],
    env: Mapping[str, str],
    dry_run: bool,
    weight_sha256: str | None = None,
    weight_display_name: str | None = None,
    vllm_version: str = "not-recorded-in-artifact",
    torch_version: str = "not-recorded-in-artifact",
) -> int:
    if not output_dirs:
        raise ConfigError("publish-existing requires at least one --output-dir")
    units = [
        load_existing_unit(
            path,
            weight_sha256=weight_sha256,
            weight_display_name=weight_display_name,
            vllm_version=vllm_version,
            torch_version=torch_version,
        )
        for path in output_dirs
    ]
    tasks_by_unit = {unit.output_dir: _unit_tasks(unit) for unit in units}
    expected_tasks = _validate_matrix(units, tasks_by_unit)
    payload = _campaign_payload(units=units, expected_tasks=expected_tasks)
    validation_campaign_id = "00000000-0000-0000-0000-000000000000"
    for unit in units:
        for task in tasks_by_unit[unit.output_dir]:
            try:
                content_digest(
                    _task_payload(
                        unit=unit,
                        task=task,
                        campaign_id=validation_campaign_id,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ExistingPublicationError(
                    f"existing task is not canonical JSON: {task['task_name']}"
                ) from error
    scoreboard_url = env.get("HELICOPTER_SCOREBOARD_URL")
    scoreboard_token = env.get("HELICOPTER_SCOREBOARD_TOKEN")
    if not scoreboard_url or not scoreboard_token:
        raise ConfigError(
            "publish-existing requires HELICOPTER_SCOREBOARD_URL and "
            "HELICOPTER_SCOREBOARD_TOKEN"
        )
    client = ScoreboardClient(scoreboard_url, scoreboard_token)
    readiness = client.preflight(
        "lm-eval",
        LM_EVAL_VERSION,
        campaign_schema=EXISTING_CAMPAIGN_SCHEMA,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "mode": "publish-existing",
                    "campaign_schema": EXISTING_CAMPAIGN_SCHEMA,
                    "evaluator": {"name": "lm-eval", "version": LM_EVAL_VERSION},
                    "output_dirs": [str(unit.output_dir) for unit in units],
                    "execution_units": [
                        {
                            "output_dir": str(unit.output_dir),
                            "weight_sha256": unit.weight_sha256,
                            "weight_display_name": unit.weight_display_name,
                            "wkv_mode": unit.wkv_mode,
                            "task_count": len(tasks_by_unit[unit.output_dir]),
                            "results_sha256": unit.results_sha256,
                            "summary_sha256": unit.summary_sha256,
                            "artifacts_sha256": unit.artifact_sha256,
                            "source_timestamp": unit.source_timestamp,
                        }
                        for unit in units
                    ],
                    "expected_task_count": len(expected_tasks),
                    "scoreboard": readiness,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt = client.create_campaign(payload, str(payload["run_key"]))
    campaign_id = receipt.get("campaign_id")
    if not isinstance(campaign_id, str):
        raise PublicationError("Scoreboard returned an invalid existing campaign id")
    try:
        campaign_id = str(uuid.UUID(campaign_id))
    except ValueError as error:
        raise PublicationError(
            "Scoreboard returned a non-canonical campaign id"
        ) from error
    expected_by_identity = {str(task["identity"]): task for task in expected_tasks}
    published = 0
    for unit in units:
        for task in tasks_by_unit[unit.output_dir]:
            task_identity = str(task["identity"])
            task_payload = _task_payload(
                unit=unit,
                task=expected_by_identity[task_identity],
                campaign_id=campaign_id,
            )
            client.publish_task(campaign_id, task_identity, task_payload)
            published += 1
    if published != len(expected_tasks):
        raise PublicationError("not every existing task was published")
    client.finalize(campaign_id, len(expected_tasks))
    print(
        f"existing campaign {campaign_id} complete; "
        f"published {published} tasks from {len(units)} output directories"
    )
    return 0
