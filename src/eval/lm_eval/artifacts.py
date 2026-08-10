from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from lm_eval.utils import handle_non_serializable

from .analysis import (
    analyze_samples,
    build_task_records,
    render_markdown,
    render_task_markdown,
)


_ANALYSIS_FILES = (
    "analysis_artifacts.json",
    "bad_cases.json",
    "error_analysis.json",
    "error_analysis.md",
)
_COUNT_FIELDS = ("samples", "scored", "correct", "incorrect", "unscored")
_STAGED_RUN_PATHS = (
    "results.json",
    "summary.json",
    "artifacts.json",
    "error_analysis.json",
    "bad_cases.json",
    "error_analysis.md",
    "benchmarks",
)


class IncrementalRunArtifacts:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self._samples_logged = False
        self._task_analyses: dict[str, Mapping[str, object]] = {}
        self._task_bad_cases: dict[str, Mapping[str, object]] = {}
        self._benchmark_artifacts: dict[str, dict[str, object]] = {}

    def mark_samples_logged(self) -> None:
        self._samples_logged = True

    def add_task(self, task_name: str, rows: object) -> None:
        self.mark_samples_logged()
        if task_name in self._task_analyses:
            raise RuntimeError(f"lm-eval samples overlap for {task_name}")
        analysis, bad_cases = analyze_samples({task_name: rows})
        benchmark_artifacts = write_benchmark_artifacts(
            self.output_dir,
            {task_name: rows},
            analysis,
        )
        if len(benchmark_artifacts) != 1:
            raise RuntimeError(f"lm-eval artifacts are incomplete for {task_name}")
        self._task_analyses[task_name] = analysis
        self._task_bad_cases[task_name] = bad_cases
        self._benchmark_artifacts[task_name] = benchmark_artifacts[0]

    def finish(self, evaluator_version: str) -> None:
        analysis_paths: dict[str, str] = {}
        if self._samples_logged:
            analysis, bad_cases = _merge_task_analyses(
                self._task_analyses,
                self._task_bad_cases,
            )
            (self.output_dir / "benchmarks").mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
            _write_analysis(self.output_dir, analysis, bad_cases)
            analysis_paths = _analysis_paths()

        write_json(
            self.output_dir / "artifacts.json",
            {
                "schema_version": 2,
                "evaluator": {"name": "lm-eval", "version": evaluator_version},
                "results_path": "results.json",
                "summary_path": "summary.json",
                "benchmark_artifacts": [
                    self._benchmark_artifacts[task_name]
                    for task_name in sorted(self._benchmark_artifacts)
                ],
                **analysis_paths,
            },
        )


def reset_run_artifacts(output_dir: Path) -> None:
    reset_analysis_artifacts(output_dir)
    _remove_managed_directory(output_dir / "samples")


def reset_analysis_artifacts(output_dir: Path) -> None:
    _remove_managed_directory(output_dir / "benchmarks")
    for name in _ANALYSIS_FILES:
        _remove_managed_file(output_dir / name)


def write_run_artifacts(
    output_dir: Path,
    results: Mapping[str, object],
    evaluator_version: str,
) -> None:
    samples = results.get("samples")
    accumulator = IncrementalRunArtifacts(output_dir)
    if samples is not None:
        if not isinstance(samples, Mapping):
            raise RuntimeError("lm-eval samples must be an object")
        accumulator.mark_samples_logged()
        for task_name, rows in sorted(samples.items()):
            if not isinstance(task_name, str):
                raise RuntimeError("lm-eval sample task names must be strings")
            accumulator.add_task(task_name, rows)
    accumulator.finish(evaluator_version)


def write_posthoc_analysis(
    *,
    output_dir: Path,
    results_path: Path,
    samples: Mapping[str, object],
    examples_per_task: int,
) -> None:
    reset_analysis_artifacts(output_dir)
    analysis, bad_cases = analyze_samples(
        samples, examples_per_task=examples_per_task
    )
    _write_analysis(output_dir, analysis, bad_cases)
    benchmark_artifacts = write_benchmark_artifacts(
        output_dir, samples, analysis
    )
    write_json(
        output_dir / "analysis_artifacts.json",
        {
            "schema_version": 1,
            "source_results_path": results_path.name,
            "source_results_sha256": _sha256(results_path),
            **_analysis_paths(),
            "benchmark_artifacts": benchmark_artifacts,
        },
    )


def write_benchmark_artifacts(
    output_dir: Path,
    samples: Mapping[str, object],
    analysis: Mapping[str, object],
) -> list[dict[str, object]]:
    summaries = _task_summaries(analysis)
    benchmark_root = output_dir / "benchmarks"
    benchmark_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    artifacts: list[dict[str, object]] = []
    if not all(isinstance(task_name, str) for task_name in samples):
        raise ValueError("sample task names must be strings")
    for task_name, rows in sorted(samples.items()):
        records = build_task_records(task_name, rows)
        errors = [
            record
            for record in records
            if record["status"] in {"incorrect", "quality_outlier"}
        ]
        task_dir = benchmark_root / _benchmark_dir_name(task_name)
        task_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        summary = {"schema_version": 1, **summaries[task_name]}
        write_json(task_dir / "summary.json", summary)
        write_jsonl(task_dir / "records.jsonl", records)
        write_jsonl(task_dir / "errors.jsonl", errors)
        write_text(
            task_dir / "report.md", render_task_markdown(summary, records)
        )
        artifacts.append(
            {
                "task_name": task_name,
                "directory": task_dir.relative_to(output_dir).as_posix(),
                "summary_path": _relative(task_dir / "summary.json", output_dir),
                "records_path": _relative(task_dir / "records.jsonl", output_dir),
                "errors_path": _relative(task_dir / "errors.jsonl", output_dir),
                "report_path": _relative(task_dir / "report.md", output_dir),
                "samples": len(records),
                "errors": len(errors),
            }
        )
    return artifacts


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            default=handle_non_serializable,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def write_spooled_results(
    path: Path,
    metadata: Mapping[str, object],
    sample_paths: Mapping[str, Path],
    *,
    samples_logged: bool,
) -> None:
    if "samples" in metadata:
        raise ValueError("spooled result metadata must not contain samples")
    if not all(isinstance(task_name, str) for task_name in sample_paths):
        raise ValueError("spooled samples must use string task names")

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    keys = set(metadata)
    if samples_logged:
        keys.add("samples")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        for field_index, name in enumerate(sorted(keys)):
            if field_index:
                stream.write(",\n")
            stream.write("  ")
            json.dump(name, stream, ensure_ascii=False)
            stream.write(": ")
            if name != "samples":
                json.dump(
                    metadata[name],
                    stream,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                continue
            stream.write("{")
            for task_index, task_name in enumerate(sorted(sample_paths)):
                stream.write("\n    " if task_index == 0 else ",\n    ")
                json.dump(task_name, stream, ensure_ascii=False)
                stream.write(": ")
                with sample_paths[task_name].open("r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
            if sample_paths:
                stream.write("  ")
            stream.write("}")
        stream.write("\n}\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def install_staged_run(staging_dir: Path, output_dir: Path) -> None:
    for name in ("results.json", "summary.json", "artifacts.json"):
        source = staging_dir / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"staged lm-eval output is missing {name}")

    reset_run_artifacts(output_dir)
    for name in _STAGED_RUN_PATHS:
        source = staging_dir / name
        if source.exists():
            source.replace(output_dir / name)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            json.dump(
                row,
                stream,
                default=handle_non_serializable,
                ensure_ascii=False,
                sort_keys=True,
            )
            stream.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _write_analysis(
    output_dir: Path,
    analysis: Mapping[str, object],
    bad_cases: Mapping[str, object],
) -> None:
    write_json(output_dir / "error_analysis.json", analysis)
    write_json(output_dir / "bad_cases.json", bad_cases)
    write_text(
        output_dir / "error_analysis.md", render_markdown(analysis, bad_cases)
    )


def _merge_task_analyses(
    task_analyses: Mapping[str, Mapping[str, object]],
    task_bad_cases: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    if not task_analyses:
        return analyze_samples({})

    totals: Counter[str] = Counter()
    family_totals: dict[str, Counter[str]] = {}
    task_summaries: list[Mapping[str, object]] = []
    cases: list[object] = []
    interpretation: object = {}
    selection: object = {}
    for task_name in sorted(task_analyses):
        analysis = task_analyses[task_name]
        summaries = analysis.get("tasks")
        if not isinstance(summaries, list) or len(summaries) != 1:
            raise RuntimeError(f"lm-eval task analysis is invalid for {task_name}")
        summary = summaries[0]
        if not isinstance(summary, Mapping):
            raise RuntimeError(f"lm-eval task analysis is invalid for {task_name}")
        task_summaries.append(summary)
        counts = Counter(
            {
                name: int(summary.get(name, 0))
                for name in _COUNT_FIELDS
            }
        )
        totals.update(counts)
        family = str(summary.get("task_family", task_name))
        family_totals.setdefault(family, Counter()).update(counts)
        interpretation = analysis.get("interpretation", interpretation)

        bad_cases = task_bad_cases.get(task_name)
        if not isinstance(bad_cases, Mapping):
            raise RuntimeError(f"lm-eval bad-case analysis is invalid for {task_name}")
        raw_cases = bad_cases.get("cases")
        if not isinstance(raw_cases, list):
            raise RuntimeError(f"lm-eval bad-case analysis is invalid for {task_name}")
        cases.extend(raw_cases)
        selection = bad_cases.get("selection", selection)

    families = [
        {
            "task_family": family,
            **{name: counts[name] for name in _COUNT_FIELDS},
            "incorrect_rate": (
                counts["incorrect"] / counts["scored"]
                if counts["scored"]
                else None
            ),
        }
        for family, counts in sorted(family_totals.items())
    ]
    analysis = {
        "schema_version": 1,
        **{name: totals[name] for name in _COUNT_FIELDS},
        "incorrect_rate": (
            totals["incorrect"] / totals["scored"]
            if totals["scored"]
            else None
        ),
        "task_families": families,
        "tasks": task_summaries,
        "interpretation": interpretation,
    }
    bad_cases = {
        "schema_version": 1,
        "selection": selection,
        "cases": cases,
    }
    return analysis, bad_cases


def _analysis_paths() -> dict[str, str]:
    return {
        "error_analysis_path": "error_analysis.json",
        "bad_cases_path": "bad_cases.json",
        "error_analysis_markdown_path": "error_analysis.md",
    }


def _task_summaries(
    analysis: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_summaries = analysis.get("tasks")
    if not isinstance(raw_summaries, list):
        raise ValueError("analysis does not contain task summaries")
    return {
        item["task_name"]: item
        for item in raw_summaries
        if isinstance(item, Mapping) and isinstance(item.get("task_name"), str)
    }


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


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _remove_managed_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlinked artifact directory: {path.name}")
    if not path.exists():
        return
    if not path.is_dir():
        raise RuntimeError(f"artifact path must be a directory: {path.name}")
    shutil.rmtree(path)


def _remove_managed_file(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"artifact path must be a file: {path.name}")
    path.unlink(missing_ok=True)
