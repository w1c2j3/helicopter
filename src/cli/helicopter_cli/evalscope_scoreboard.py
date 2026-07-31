"""Persist official EvalScope Agent results in the local scoreboard database.

EvalScope writes JSONL prediction/review artifacts because its official scorers
need them while a run is in progress.  This module treats those artifacts as
an import stream: the official review is copied verbatim into the completion
context, the official sample score is copied into ``eval``, and the official
report is copied into ``scores.metrics``.  No answer is repaired or rescored
here.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


@dataclass(frozen=True)
class EvalScopeImportPlan:
    benchmark: str
    model_name: str
    work_dir: Path
    report: dict[str, Any]
    completion_payloads: list[dict[str, Any]]
    eval_payloads: list[dict[str, Any]]
    context_audit: dict[str, Any]
    missing_reviews: int
    invalid_reviews: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"__jsonl_error__": f"invalid JSONL at line {line_number}", "index": line_number - 1})
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _model_dir(work_dir: Path, model_name: str, kind: str) -> Path:
    root = work_dir / kind
    exact = root / model_name
    if exact.is_dir():
        return exact
    candidates = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"EvalScope {kind} model directory not found for {model_name}: {root}")


def _load_report(work_dir: Path, model_name: str, benchmark: str) -> dict[str, Any]:
    path = work_dir / "reports" / model_name / f"{benchmark}.json"
    if not path.is_file():
        report_dir = work_dir / "reports"
        candidates = sorted(report_dir.glob(f"*/{benchmark}.json")) if report_dir.is_dir() else []
        if len(candidates) == 1:
            path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"EvalScope official report not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EvalScope official report must be a JSON object: {path}")
    return value


def _model_output_parts(prediction: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    output = prediction.get("model_output")
    if not isinstance(output, Mapping):
        return "", None, None
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return "", str(output.get("error") or "") or None, None
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else ""
    finish_reason = choice.get("finish_reason") or choice.get("stop_reason") or output.get("stop_reason")
    return str(content or ""), str(output.get("error") or "") or None, str(finish_reason or "") or None


def _context_audit(prediction: Mapping[str, Any]) -> dict[str, Any]:
    content, error, finish_reason = _model_output_parts(prediction)
    output = prediction.get("model_output")
    usage = output.get("usage") if isinstance(output, Mapping) else None
    perf = output.get("perf_metrics") if isinstance(output, Mapping) else None
    choice = None
    choices = output.get("choices") if isinstance(output, Mapping) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
    finish = str(finish_reason or "").casefold()
    error_text = " ".join(str(item or "") for item in (error, output.get("error") if isinstance(output, Mapping) else None)).casefold()
    context_markers = ("context length", "maximum context", "context window", "too many tokens", "token limit", "input too long")
    context_error = any(marker in error_text for marker in context_markers) or finish in {"length", "max_tokens"}
    input_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    if input_tokens is None and isinstance(perf, Mapping):
        input_tokens = perf.get("input_tokens")
    output_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    if output_tokens is None and isinstance(perf, Mapping):
        output_tokens = perf.get("output_tokens")
    return {
        "status": "context_error" if context_error else "ok",
        "context_error": context_error,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "content_chars": len(content),
        "error": error,
        "message_count": len(prediction.get("messages", [])) if isinstance(prediction.get("messages"), list) else None,
        "has_tools": bool(prediction.get("metadata", {}).get("function")) if isinstance(prediction.get("metadata"), Mapping) else False,
        "choice_keys": sorted(choice.keys()) if isinstance(choice, Mapping) else [],
    }


def _sample_score(review: Mapping[str, Any]) -> tuple[float | None, str | None, Any, str]:
    sample_score = review.get("sample_score")
    if not isinstance(sample_score, Mapping):
        return None, None, None, "official review has no sample_score"
    score = sample_score.get("score")
    value = score.get("value") if isinstance(score, Mapping) else None
    raw_acc = value.get("acc") if isinstance(value, Mapping) else None
    try:
        acc = float(raw_acc) if raw_acc is not None else None
    except (TypeError, ValueError):
        acc = None
    extracted = score.get("extracted_prediction") if isinstance(score, Mapping) else None
    if extracted is None and isinstance(score, Mapping):
        extracted = score.get("prediction")
    metadata = sample_score.get("sample_metadata")
    error = score.get("explanation") if isinstance(score, Mapping) else None
    score_metadata = score.get("metadata") if isinstance(score, Mapping) else None
    if not error and isinstance(score_metadata, Mapping):
        error = score_metadata.get("error_message") or score_metadata.get("error")
    if acc is None:
        error = error or "official review has no numeric acc"
    return acc, str(extracted) if extracted is not None else None, metadata, str(error or "")


def _ground_truth(prediction: Mapping[str, Any], review: Mapping[str, Any]) -> Any:
    for source in (review, prediction):
        score = source.get("sample_score") if isinstance(source, Mapping) else None
        metadata = score.get("sample_metadata") if isinstance(score, Mapping) else None
        if isinstance(metadata, Mapping) and "ground_truth" in metadata:
            return metadata["ground_truth"]
        metadata = source.get("metadata") if isinstance(source, Mapping) else None
        if isinstance(metadata, Mapping) and "ground_truth" in metadata:
            return metadata["ground_truth"]
    return None


def _subset_name(path: Path, benchmark: str) -> str:
    return path.stem.removeprefix(f"{benchmark}_") or benchmark


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_import_plan(work_dir: Path, *, model_name: str, benchmark: str = "bfcl_v4") -> EvalScopeImportPlan:
    """Build DB payloads without importing scoreboard dependencies.

    The prediction/review records are retained in ``agent_result`` so the DB
    remains the audit source after temporary JSON artifacts are removed.
    """

    work_dir = Path(work_dir)
    report = _load_report(work_dir, model_name, benchmark)
    prediction_dir = _model_dir(work_dir, model_name, "predictions")
    review_dir = _model_dir(work_dir, model_name, "reviews")
    completion_payloads: list[dict[str, Any]] = []
    eval_payloads: list[dict[str, Any]] = []
    audit_counts: Counter[str] = Counter()
    missing_reviews = 0
    invalid_reviews = 0
    sample_index = 0
    prediction_paths = sorted(prediction_dir.glob("*.jsonl"))
    if not prediction_paths:
        raise FileNotFoundError(f"no EvalScope prediction JSONL files found: {prediction_dir}")
    for prediction_path in prediction_paths:
        review_path = review_dir / prediction_path.name
        predictions = _read_jsonl(prediction_path)
        reviews = _read_jsonl(review_path)
        if not review_path.is_file():
            missing_reviews += len(predictions)
        for ordinal, prediction in enumerate(predictions):
            review = reviews[ordinal] if ordinal < len(reviews) else {}
            audit = _context_audit(prediction)
            audit_counts[str(audit["status"])] += 1
            acc, extracted, metadata, reason = _sample_score(review)
            if not review:
                missing_reviews += 1
            if acc is None:
                invalid_reviews += 1
            answer = extracted
            if answer is None:
                answer, _error, _finish = _model_output_parts(prediction)
            ref_answer = _ground_truth(prediction, review)
            subset = _subset_name(prediction_path, benchmark)
            agent_result = {
                "prediction": prediction,
                "review": review,
                "official_sample_score": acc,
                "official_extracted_prediction": extracted,
                "official_reference": ref_answer,
                "subset": subset,
                "source_index": ordinal,
                "context_audit": audit,
            }
            completion_payloads.append(
                {
                    "_stage": "answer",
                    "sample_index": sample_index,
                    "repeat_index": 0,
                    "pass_index": 0,
                    "status": "Completed" if acc is not None else "Failed",
                    "sampling_config": {"cot_mode": "NoCoT", "source": "evalscope_official", "benchmark": benchmark},
                    "agent_result": agent_result,
                }
            )
            if acc is not None:
                eval_payloads.append(
                    {
                        "sample_index": sample_index,
                        "repeat_index": 0,
                        "pass_index": 0,
                        "answer": answer,
                        "ref_answer": ref_answer,
                        "is_passed": acc == 1.0,
                        "fail_reason": "" if acc == 1.0 else reason,
                    }
                )
            sample_index += 1
    context_audit = {
        "samples": sample_index,
        "status_counts": dict(sorted(audit_counts.items())),
        "context_error_samples": audit_counts.get("context_error", 0),
        "missing_reviews": missing_reviews,
        "invalid_reviews": invalid_reviews,
    }
    return EvalScopeImportPlan(
        benchmark=benchmark,
        model_name=model_name,
        work_dir=work_dir,
        report=report,
        completion_payloads=completion_payloads,
        eval_payloads=eval_payloads,
        context_audit=context_audit,
        missing_reviews=missing_reviews,
        invalid_reviews=invalid_reviews,
    )


async def persist_import_plan(
    plan: EvalScopeImportPlan,
    *,
    root: Path,
    job_name: str = "evalscope_bfcl_v4_official",
) -> str:
    """Persist one official run into the existing scoreboard database."""

    scoreboard_path = Path(root) / "src" / "scoreboard-server"
    if str(scoreboard_path) not in sys.path:
        sys.path.insert(0, str(scoreboard_path))
    from scoreboard_server.db.connection import close_db, init_db
    from scoreboard_server.db.repository import ScoreboardStore
    from scoreboard_server.db.settings import DatabaseSettings

    settings = DatabaseSettings.from_env()
    await init_db(settings, generate_schemas=False)
    try:
        store = ScoreboardStore(settings=settings)
        sampling_config = {
            "cot_mode": "NoCoT",
            "source": "evalscope_official",
            "benchmark": plan.benchmark,
            "work_dir": str(plan.work_dir),
            "context_audit": plan.context_audit,
        }
        config_path = str(plan.work_dir / "configs" / "task_config.yaml")
        task_id = await store.get_or_create_task(
            job_name=job_name,
            job_id=None,
            dataset=plan.benchmark,
            model=plan.model_name,
            is_param_search=False,
            sampling_config=sampling_config,
            config_path=config_path if Path(config_path).is_file() else None,
            allow_resume=True,
        )
        await store.ensure_benchmark_num_samples(
            dataset=plan.benchmark,
            num_samples=int(plan.context_audit["samples"]),
        )
        inserted, inserted_evals = await store.insert_completion_eval_payloads_bulk(
            completion_payloads=plan.completion_payloads,
            eval_payloads=plan.eval_payloads,
            task_id=task_id,
        )
        metrics = {
            "score": plan.report.get("score"),
            "official_report": plan.report,
            "context_audit": plan.context_audit,
            "imported_completions": inserted,
            "imported_evals": inserted_evals,
            "work_dir": str(plan.work_dir),
        }
        await store.record_score_payload(
            task_id=task_id,
            payload={"cot_mode": "NoCoT", "metrics": metrics},
            mark_completed=not plan.invalid_reviews,
        )
        if plan.invalid_reviews:
            await store.update_task_status(task_id=task_id, status="Failed")
        return f"task={task_id} completions={inserted} evals={inserted_evals} score={plan.report.get('score')}"
    finally:
        await close_db()


def persist_import_plan_sync(plan: EvalScopeImportPlan, *, root: Path, job_name: str = "evalscope_bfcl_v4_official") -> str:
    return asyncio.run(persist_import_plan(plan, root=root, job_name=job_name))


def cleanup_json_artifacts(work_dir: Path) -> int:
    """Remove only EvalScope JSON/JSONL artifacts after a verified DB import."""

    removed = 0
    for path in sorted(Path(work_dir).rglob("*")):
        if path.is_file() and path.suffix.casefold() in {".json", ".jsonl"}:
            path.unlink()
            removed += 1
    return removed
