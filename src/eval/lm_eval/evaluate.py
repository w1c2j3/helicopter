from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

from helicopter_lighteval.http_pool import PoolError, VLLMHttpPool
from helicopter_lighteval.publish import PublicationError

from .analysis import (
    analyze_samples,
    build_task_records,
    render_markdown,
    render_task_markdown,
)
from .config import ConfigError, LMEvalConfig
from .model import RWKVVLLMHttpLM


@contextmanager
def _remote_dataset_code(enabled: bool):
    import datasets.config

    previous = datasets.config.HF_DATASETS_TRUST_REMOTE_CODE
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = enabled
    try:
        yield
    finally:
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = previous


def _evaluation_task_specs(manager, tasks: Sequence[str], override: str | None):
    if override is None:
        return list(tasks)
    specs: list[object] = []
    for task_name in tasks:
        entry = manager.task_index.get(task_name)
        if entry is None:
            raise ConfigError(f"unknown task for dataset_path_override: {task_name}")
        built = manager._factory.build(
            entry,
            overrides={"dataset_path": override},
            registry=manager.task_index,
        )
        if isinstance(built, list):
            specs.extend(built)
        else:
            specs.append(built)
    return specs


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    pools: list[VLLMHttpPool] = []
    try:
        config = LMEvalConfig.read(config_path, env)
        task_manager, resolved_tasks = _resolve_tasks(
            config.tasks, config.task_include_paths
        )
        benchmark_runs = _benchmark_runs(config, task_manager, resolved_tasks)
        version = importlib.metadata.version("lm-eval")
        if version != "0.4.12":
            raise ConfigError(f"lm-eval version must be 0.4.12, found {version}")
        client = None
        campaign = None
        expected_tasks = None
        scoreboard = None
        if config.publish:
            from helicopter_lighteval.publish import ScoreboardClient
            from .publish import campaign_payload, expected_tasks as build_expected
            from .publish import task_metadata

            if config.scoreboard_url is None or config.scoreboard_token is None:
                raise ConfigError("published evaluation requires Scoreboard access")
            metadata = task_metadata(task_manager, config.tasks, resolved_tasks)
            expected_tasks = build_expected(config, metadata)
            campaign = campaign_payload(config, metadata, expected_tasks)
            client = ScoreboardClient(config.scoreboard_url, config.scoreboard_token)
            scoreboard = client.preflight("lm-eval", version)
        readiness: list[dict[str, object]] = []
        for unit in config.execution_units:
            pool = VLLMHttpPool(unit.manifest)
            pools.append(pool)
            model_id = pool.preflight()
            readiness.append(
                {
                    "model_id": model_id,
                    "weight_sha256": unit.weight_sha256,
                    "wkv_mode": unit.wkv_mode,
                    "global_step": unit.manifest.global_step,
                    "max_model_len": unit.manifest.max_model_len,
                    "effective_max_length": unit.manifest.max_model_len - 2,
                }
            )
        if dry_run:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "evaluator": {"name": "lm-eval", "version": version},
                        "config": config.public(),
                        **(readiness[0] if len(readiness) == 1 else {}),
                        "execution_units": readiness,
                        "resolved_tasks": list(resolved_tasks),
                        "task_selection": {
                            "selectors": list(config.tasks),
                            "resolved": _describe_tasks(
                                task_manager, resolved_tasks
                            ),
                            "benchmark_runs": [
                                {
                                    "selector": benchmark.selector,
                                    "resolved_tasks": list(tasks),
                                    "config": benchmark.public(),
                                }
                                for benchmark, tasks in benchmark_runs
                                if benchmark is not None
                            ],
                        },
                        "expected_task_count": (
                            len(expected_tasks) if expected_tasks is not None else 0
                        ),
                        "scoreboard": scoreboard,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        import lm_eval

        campaign_id = None
        if config.publish:
            if client is None or campaign is None or expected_tasks is None:
                raise ConfigError("published evaluation was not initialized")
            receipt = client.create_campaign(campaign, str(campaign["run_key"]))
            try:
                campaign_id = str(uuid.UUID(str(receipt["campaign_id"])))
            except ValueError as error:
                raise PublicationError(
                    "Scoreboard returned an invalid campaign id"
                ) from error

        completed = 0
        for index, (unit, pool, ready) in enumerate(
            zip(config.execution_units, pools, readiness, strict=True)
        ):
            result_parts: list[Mapping[str, object]] = []
            for benchmark, run_tasks in benchmark_runs:
                batch_size = benchmark.batch_size if benchmark else config.batch_size
                max_gen_toks = (
                    benchmark.max_gen_toks if benchmark else config.max_gen_toks
                )
                limit = benchmark.limit if benchmark else config.limit
                prompt = benchmark.prompt if benchmark else config.prompt
                generation_kwargs = (
                    benchmark.generation_kwargs
                    if benchmark
                    else config.generation_kwargs
                )
                model = RWKVVLLMHttpLM(
                    pool=pool,
                    eot_token_id=config.eot_token_id,
                    batch_size=batch_size,
                    max_gen_toks=max_gen_toks,
                    prompt_profile=prompt.profile,
                    generation_prompt=prompt.generation_prompt,
                )
                with _remote_dataset_code(
                    benchmark.trust_remote_dataset_code if benchmark else False
                ):
                    part = lm_eval.simple_evaluate(
                        model=model,
                        tasks=_evaluation_task_specs(
                            task_manager,
                            run_tasks,
                            benchmark.dataset_path_override if benchmark else None,
                        ),
                        batch_size=batch_size,
                        limit=limit,
                        log_samples=config.log_samples or config.publish,
                        task_manager=task_manager,
                        num_fewshot=prompt.num_fewshot,
                        system_instruction=prompt.system_instruction,
                        apply_chat_template=prompt.apply_chat_template,
                        fewshot_as_multiturn=prompt.fewshot_as_multiturn,
                        gen_kwargs=(
                            dict(generation_kwargs) if generation_kwargs else None
                        ),
                        confirm_run_unsafe_code=(
                            benchmark.confirm_run_unsafe_code if benchmark else False
                        ),
                    )
                if part is None:
                    selector = benchmark.selector if benchmark else ",".join(config.tasks)
                    raise RuntimeError(f"lm-eval returned no results for {selector}")
                result_parts.append(part)
            results = _merge_results(result_parts)
            output_dir = _unit_output_dir(
                config.output_dir,
                unit,
                index,
                total_units=len(config.execution_units),
            )
            _write_results(
                config=config,
                model_id=str(ready["model_id"]),
                version=version,
                results=results,
                output_dir=output_dir,
                manifest=unit.manifest,
                weight_sha256=unit.weight_sha256,
            )
            benchmark_logs = output_dir / "benchmarks"
            if benchmark_logs.is_dir():
                print(f"lm-eval benchmark logs written to {benchmark_logs}")
            if campaign_id is not None:
                from .publish import publish_unit

                unit_expected = [
                    task
                    for task in expected_tasks
                    if task["weight_sha256"] == unit.weight_sha256
                    and task["wkv_mode"] == unit.wkv_mode
                ]
                completed += publish_unit(
                    output_dir=output_dir,
                    campaign_id=campaign_id,
                    expected=unit_expected,
                    unit=unit,
                    config=config,
                    client=client,
                )
        if campaign_id is not None:
            if completed != len(expected_tasks):
                raise PublicationError(
                    "not every expected evaluation task was published"
                )
            client.finalize(campaign_id, len(expected_tasks))
        print(f"lm-eval results written to {config.output_dir}")
        return 0
    except (ConfigError, PoolError, PublicationError) as error:
        raise SystemExit(str(error)) from error
    except Exception as error:
        name = f"{type(error).__module__}.{type(error).__qualname__}"
        detail = str(error).strip()
        suffix = f": {detail}" if detail else ""
        raise SystemExit(f"lm-eval evaluation failed: {name}{suffix}") from error
    finally:
        for pool in pools:
            pool.close()


def _unit_output_dir(
    configured: Path, unit, index: int, *, total_units: int
) -> Path:
    if total_units == 1:
        return configured
    digest = unit.weight_sha256 or f"unidentified-{index}"
    return configured / digest / unit.wkv_mode


def _resolve_tasks(tasks: tuple[str, ...], include_paths: tuple[Path, ...] = ()):
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=list(include_paths) or None)
    resolved: list[str] = []
    for selector in tasks:
        matches = manager.match_tasks([selector])
        if not matches:
            raise ConfigError(f"unknown lm-eval task, group, tag, or pattern: {selector}")
        for task_name in matches:
            if task_name not in resolved:
                resolved.append(task_name)
    return manager, tuple(resolved)


def _benchmark_runs(config: LMEvalConfig, manager, resolved_tasks: tuple[str, ...]):
    if not config.benchmarks:
        return ((None, resolved_tasks),)
    runs = []
    claimed: dict[str, str] = {}
    for benchmark in config.benchmarks:
        matches = tuple(manager.match_tasks([benchmark.selector]))
        if not matches:
            raise ConfigError(
                f"unknown lm-eval benchmark selector: {benchmark.selector}"
            )
        for task_name in matches:
            previous = claimed.get(task_name)
            if previous is not None:
                raise ConfigError(
                    f"benchmark selectors {previous} and {benchmark.selector} "
                    f"both resolve to {task_name}"
                )
            claimed[task_name] = benchmark.selector
        runs.append((benchmark, matches))
    if tuple(claimed) != resolved_tasks:
        raise ConfigError("benchmark config resolution does not match configured tasks")
    return tuple(runs)


_MERGED_RESULT_FIELDS = {
    "configs",
    "group_subtasks",
    "groups",
    "higher_is_better",
    "n-samples",
    "n-shot",
    "results",
    "samples",
    "versions",
}


def _merge_results(parts: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not parts:
        raise RuntimeError("lm-eval returned no benchmark result parts")
    if len(parts) == 1:
        return parts[0]
    merged: dict[str, object] = {}
    for part in parts:
        for name, value in part.items():
            if name not in _MERGED_RESULT_FIELDS:
                merged.setdefault(name, value)
                continue
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise RuntimeError(f"lm-eval result field {name} must be an object")
            destination = merged.setdefault(name, {})
            if not isinstance(destination, dict):
                raise RuntimeError(f"lm-eval result field {name} cannot be merged")
            overlaps = sorted(set(destination).intersection(value))
            if overlaps:
                raise RuntimeError(
                    f"lm-eval benchmark results overlap in {name}: "
                    + ", ".join(overlaps)
                )
            destination.update(value)
    return merged


def _describe_tasks(manager, tasks: tuple[str, ...]) -> list[dict[str, object]]:
    descriptions: list[dict[str, object]] = []
    for task_name in tasks:
        entry = manager.task_index[task_name]
        config = entry.cfg or {}
        descriptions.append(
            {
                "name": task_name,
                "kind": entry.kind.name.lower(),
                "output_type": config.get("output_type"),
            }
        )
    return descriptions


def _write_results(
    *,
    config: LMEvalConfig,
    model_id: str,
    version: str,
    results: Mapping[str, object],
    output_dir: Path | None = None,
    manifest=None,
    weight_sha256: str | None = None,
) -> None:
    from lm_eval.utils import handle_non_serializable

    destination = output_dir or config.output_dir
    execution_manifest = manifest or config.manifest
    _prepare_output_dir(destination)
    raw = json.loads(
        json.dumps(results, default=handle_non_serializable, ensure_ascii=False)
    )
    summary = {
        "schema_version": 1,
        "evaluator": {"name": "lm-eval", "version": version},
        "backend": "vllm_http",
        "model_id": model_id,
        "weight_sha256": weight_sha256,
        "global_step": execution_manifest.global_step,
        "wkv_mode": execution_manifest.wkv_mode,
        "max_model_len": execution_manifest.max_model_len,
        "effective_max_length": execution_manifest.max_model_len - 2,
        "tasks": list(config.tasks),
        "prompt": config.prompt.public(),
        "generation_kwargs": dict(config.generation_kwargs),
        "benchmark_configs": [
            benchmark.public() for benchmark in config.benchmarks
        ],
        "task_include_paths": [str(path) for path in config.task_include_paths],
        "metrics": raw.get("results", {}),
        "versions": raw.get("versions", {}),
    }
    _atomic_json(destination / "results.json", raw)
    _atomic_json(destination / "summary.json", summary)
    _write_sample_artifacts(destination, raw, version)


def _write_sample_artifacts(
    output_dir: Path,
    results: Mapping[str, object],
    version: str,
) -> None:
    raw_samples = results.get("samples")
    artifacts: list[dict[str, object]] = []
    benchmark_artifacts: list[dict[str, object]] = []
    analysis_paths: dict[str, str] = {}
    if isinstance(raw_samples, Mapping):
        samples_dir = output_dir / "samples"
        _prepare_output_dir(samples_dir)
        for index, (task_name, rows) in enumerate(sorted(raw_samples.items())):
            if not isinstance(task_name, str) or not isinstance(rows, list):
                raise RuntimeError("lm-eval samples must map task names to arrays")
            path = samples_dir / f"{index:04d}.json"
            _atomic_json(path, rows)
            artifacts.append(
                {
                    "task_name": task_name,
                    "path": path.relative_to(output_dir).as_posix(),
                    "samples": len(rows),
                }
            )
        analysis, bad_cases = analyze_samples(raw_samples)
        _atomic_json(output_dir / "error_analysis.json", analysis)
        _atomic_json(output_dir / "bad_cases.json", bad_cases)
        _atomic_text(
            output_dir / "error_analysis.md",
            render_markdown(analysis, bad_cases),
        )
        analysis_paths = {
            "error_analysis_path": "error_analysis.json",
            "bad_cases_path": "bad_cases.json",
            "error_analysis_markdown_path": "error_analysis.md",
        }
        summaries = {
            item["task_name"]: item
            for item in analysis["tasks"]
            if isinstance(item, Mapping) and isinstance(item.get("task_name"), str)
        }
        benchmarks_dir = output_dir / "benchmarks"
        _prepare_output_dir(benchmarks_dir)
        for task_name, rows in sorted(raw_samples.items()):
            if not isinstance(task_name, str):
                raise RuntimeError("lm-eval sample task names must be strings")
            records = build_task_records(task_name, rows)
            errors = [
                record
                for record in records
                if record["status"] in {"incorrect", "quality_outlier"}
            ]
            task_dir = benchmarks_dir / _benchmark_dir_name(task_name)
            _prepare_output_dir(task_dir)
            summary = {"schema_version": 1, **summaries[task_name]}
            _atomic_json(task_dir / "summary.json", summary)
            _atomic_jsonl(task_dir / "records.jsonl", records)
            _atomic_jsonl(task_dir / "errors.jsonl", errors)
            _atomic_text(
                task_dir / "report.md",
                render_task_markdown(summary, records),
            )
            benchmark_artifacts.append(
                {
                    "task_name": task_name,
                    "directory": task_dir.relative_to(output_dir).as_posix(),
                    "summary_path": (task_dir / "summary.json")
                    .relative_to(output_dir)
                    .as_posix(),
                    "records_path": (task_dir / "records.jsonl")
                    .relative_to(output_dir)
                    .as_posix(),
                    "errors_path": (task_dir / "errors.jsonl")
                    .relative_to(output_dir)
                    .as_posix(),
                    "report_path": (task_dir / "report.md")
                    .relative_to(output_dir)
                    .as_posix(),
                    "samples": len(records),
                    "errors": len(errors),
                }
            )
    _atomic_json(
        output_dir / "artifacts.json",
        {
            "schema_version": 1,
            "evaluator": {"name": "lm-eval", "version": version},
            "results_path": "results.json",
            "summary_path": "summary.json",
            "sample_artifacts": artifacts,
            "benchmark_artifacts": benchmark_artifacts,
            **analysis_paths,
        },
    )


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.geteuid()
    ):
        raise ConfigError("output_dir must be an owned regular directory")
    path.chmod(0o700)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    value = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, value)


def _benchmark_dir_name(task_name: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in task_name
    ).strip(".")
    if not safe:
        safe = "task"
    if safe != task_name:
        suffix = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}--{suffix}"
    return safe
