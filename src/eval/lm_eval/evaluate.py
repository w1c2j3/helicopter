from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

from helicopter_lighteval.http_pool import PoolError, VLLMHttpPool
from helicopter_lighteval.publish import PublicationError

from .artifacts import (
    IncrementalRunArtifacts,
    install_staged_run,
    reset_run_artifacts,
    write_json,
    write_run_artifacts,
    write_spooled_results,
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


def _evaluation_task_specs(
    manager,
    tasks: Sequence[str],
    dataset_path_override: str | None,
    dataset_kwargs_override: Mapping[str, object] | None,
):
    return [
        spec
        for batch in _evaluation_task_batches(
            manager,
            tasks,
            dataset_path_override,
            dataset_kwargs_override,
        )
        for spec in batch
    ]


def _evaluation_task_batches(
    manager,
    tasks: Sequence[str],
    dataset_path_override: str | None,
    dataset_kwargs_override: Mapping[str, object] | None,
):
    if dataset_path_override is None and dataset_kwargs_override is None:
        yield list(tasks)
        return
    overrides: dict[str, object] = {}
    if dataset_path_override is not None:
        overrides["dataset_path"] = dataset_path_override
    if dataset_kwargs_override is not None:
        dataset_kwargs = dict(dataset_kwargs_override)
        features = dataset_kwargs.get("features")
        if isinstance(features, Mapping):
            from datasets import Features

            dataset_kwargs["features"] = Features.from_dict(dict(features))
        overrides["dataset_kwargs"] = dataset_kwargs
    for task_name in tasks:
        entry = manager.task_index.get(task_name)
        if entry is None:
            raise ConfigError(f"unknown task for benchmark override: {task_name}")
        split_members = _split_scoped_group_members(manager, entry, overrides)
        if split_members is not None:
            for member, child_entry, test_split in split_members:
                child = manager._factory.build(
                    child_entry,
                    overrides={"task": member, **overrides},
                    registry=manager.task_index,
                )
                dataset = getattr(child, "dataset", None)
                if not isinstance(dataset, Mapping) or test_split not in dataset:
                    raise ConfigError(
                        f"group task {member} did not load expected split {test_split}"
                    )
                child.dataset = {test_split: dataset[test_split]}
                del dataset
                yield [child]
                del child
                gc.collect()
            continue
        built = manager._factory.build(
            entry,
            overrides=overrides,
            registry=manager.task_index,
        )
        if isinstance(built, list):
            yield built
        else:
            yield [built]


def _split_scoped_group_members(manager, entry, overrides: Mapping[str, object]):
    entry_config = getattr(entry, "cfg", None)
    dataset_kwargs = overrides.get("dataset_kwargs")
    data_files = (
        dataset_kwargs.get("data_files")
        if isinstance(dataset_kwargs, Mapping)
        else None
    )
    members = entry_config.get("task") if isinstance(entry_config, Mapping) else None
    aggregate_metrics = (
        entry_config.get("aggregate_metric_list")
        if isinstance(entry_config, Mapping)
        else None
    )
    if (
        not isinstance(data_files, Mapping)
        or not isinstance(members, list)
        or aggregate_metrics
    ):
        return None
    if not members or not all(isinstance(member, str) for member in members):
        return None

    child_entries = []
    for member in members:
        child_entry = manager.task_index.get(member)
        child_config = getattr(child_entry, "cfg", None)
        test_split = (
            child_config.get("test_split")
            if isinstance(child_config, Mapping)
            else None
        )
        if (
            child_entry is None
            or not isinstance(test_split, str)
            or test_split not in data_files
        ):
            return None
        child_entries.append((member, child_entry, test_split))
    return child_entries


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
            output_dir = _unit_output_dir(
                config.output_dir,
                unit,
                index,
                total_units=len(config.execution_units),
            )
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".{output_dir.name}.spool-",
                dir=output_dir.parent,
            ) as temporary:
                staging_dir = Path(temporary)
                staging_dir.chmod(0o700)
                spool = _ResultSpool(staging_dir)
                for benchmark, run_tasks in benchmark_runs:
                    batch_size = (
                        benchmark.batch_size if benchmark else config.batch_size
                    )
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
                    task_batches = _evaluation_task_batches(
                        task_manager,
                        run_tasks,
                        benchmark.dataset_path_override if benchmark else None,
                        benchmark.dataset_kwargs_override if benchmark else None,
                    )
                    for task_specs in task_batches:
                        with _remote_dataset_code(
                            benchmark.trust_remote_dataset_code
                            if benchmark
                            else False
                        ):
                            part = lm_eval.simple_evaluate(
                                model=model,
                                tasks=task_specs,
                                batch_size=batch_size,
                                limit=limit,
                                log_samples=config.log_samples or config.publish,
                                task_manager=task_manager,
                                num_fewshot=prompt.num_fewshot,
                                system_instruction=prompt.system_instruction,
                                apply_chat_template=prompt.apply_chat_template,
                                fewshot_as_multiturn=prompt.fewshot_as_multiturn,
                                gen_kwargs=(
                                    dict(generation_kwargs)
                                    if generation_kwargs
                                    else None
                                ),
                                confirm_run_unsafe_code=(
                                    benchmark.confirm_run_unsafe_code
                                    if benchmark
                                    else False
                                ),
                            )
                        if part is None:
                            selector = (
                                benchmark.selector
                                if benchmark
                                else ",".join(config.tasks)
                            )
                            raise RuntimeError(
                                f"lm-eval returned no results for {selector}"
                            )
                        spool.add(part)
                        del part
                        gc.collect()
                _write_spooled_output(
                    config=config,
                    model_id=str(ready["model_id"]),
                    version=version,
                    spool=spool,
                    staging_dir=staging_dir,
                    manifest=unit.manifest,
                    weight_sha256=unit.weight_sha256,
                )
                _prepare_output_dir(output_dir)
                install_staged_run(staging_dir, output_dir)
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
                        result_metadata=spool.metadata,
                        sample_paths=spool.sample_paths,
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


class _ResultSpool:
    def __init__(self, staging_dir: Path) -> None:
        self.staging_dir = staging_dir
        self.metadata: dict[str, object] = {}
        self.sample_paths: dict[str, Path] = {}
        self.samples_logged = False
        self._sample_root = staging_dir / ".sample-spool"
        self._artifacts = IncrementalRunArtifacts(staging_dir)

    def add(self, part: Mapping[str, object]) -> None:
        for name, value in part.items():
            if name not in _MERGED_RESULT_FIELDS:
                self.metadata.setdefault(name, value)
                continue
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise RuntimeError(f"lm-eval result field {name} must be an object")
            if name == "samples":
                self._add_samples(value)
                continue
            destination = self.metadata.setdefault(name, {})
            if not isinstance(destination, dict):
                raise RuntimeError(f"lm-eval result field {name} cannot be merged")
            overlaps = sorted(set(destination).intersection(value))
            if overlaps:
                raise RuntimeError(
                    f"lm-eval benchmark results overlap in {name}: "
                    + ", ".join(overlaps)
                )
            destination.update(value)

    def finish(self, evaluator_version: str) -> None:
        self._artifacts.finish(evaluator_version)

    def _add_samples(self, samples: Mapping[object, object]) -> None:
        self.samples_logged = True
        self._artifacts.mark_samples_logged()
        overlaps = sorted(set(self.sample_paths).intersection(samples))
        if overlaps:
            raise RuntimeError(
                "lm-eval benchmark results overlap in samples: "
                + ", ".join(str(name) for name in overlaps)
            )
        self._sample_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for task_name, rows in sorted(samples.items()):
            if not isinstance(task_name, str):
                raise RuntimeError("lm-eval sample task names must be strings")
            sample_path = self._sample_root / f"{len(self.sample_paths):04d}.json"
            write_json(sample_path, rows)
            self._artifacts.add_task(task_name, rows)
            self.sample_paths[task_name] = sample_path


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
    destination = output_dir or config.output_dir
    execution_manifest = manifest or config.manifest
    _prepare_output_dir(destination)
    reset_run_artifacts(destination)
    summary = _result_summary(
        config=config,
        model_id=model_id,
        version=version,
        results=results,
        manifest=execution_manifest,
        weight_sha256=weight_sha256,
    )
    write_json(destination / "results.json", results)
    write_json(destination / "summary.json", summary)
    write_run_artifacts(destination, results, version)


def _write_spooled_output(
    *,
    config: LMEvalConfig,
    model_id: str,
    version: str,
    spool: _ResultSpool,
    staging_dir: Path,
    manifest,
    weight_sha256: str | None,
) -> None:
    write_spooled_results(
        staging_dir / "results.json",
        spool.metadata,
        spool.sample_paths,
        samples_logged=spool.samples_logged,
    )
    summary = _result_summary(
        config=config,
        model_id=model_id,
        version=version,
        results=spool.metadata,
        manifest=manifest,
        weight_sha256=weight_sha256,
    )
    write_json(staging_dir / "summary.json", summary)
    spool.finish(version)


def _result_summary(
    *,
    config: LMEvalConfig,
    model_id: str,
    version: str,
    results: Mapping[str, object],
    manifest,
    weight_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator": {"name": "lm-eval", "version": version},
        "backend": "vllm_http",
        "model_id": model_id,
        "weight_sha256": weight_sha256,
        "global_step": manifest.global_step,
        "wkv_mode": manifest.wkv_mode,
        "max_model_len": manifest.max_model_len,
        "effective_max_length": manifest.max_model_len - 2,
        "tasks": list(config.tasks),
        "prompt": config.prompt.public(),
        "generation_kwargs": dict(config.generation_kwargs),
        "benchmark_configs": [
            benchmark.public() for benchmark in config.benchmarks
        ],
        "task_include_paths": [str(path) for path in config.task_include_paths],
        "metrics": results.get("results", {}),
        "versions": results.get("versions", {}),
    }


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
