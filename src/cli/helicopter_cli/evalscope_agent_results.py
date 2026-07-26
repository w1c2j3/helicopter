"""Traceable extraction and discrimination for EvalScope Agent outputs.

The official EvalScope scorer remains the source of truth for benchmark
scores.  These helpers provide a strict, local diagnostic layer: an invalid
format is not converted into an answer, and a missing answer is never filled
from the reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .lighteval_answer_adapters import extract_choice_answer, extract_math_answer


_SWE_BENCH_BLOCK = re.compile(r"```mswea_bash_command\n(.*?)\n```", re.DOTALL)
_ANSWER_LINE = re.compile(r"(?im)^\s*(?:final\s+answer|exact\s+answer)\s*[:：]\s*(\S.*?)(?:\r?\n|$)")
_CODE_BLOCK = re.compile(r"(?ms)^\s*```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)\n\s*```\s*$")


@dataclass(frozen=True)
class ExtractionResult:
    format_kind: str
    raw_response: str
    extracted_answer: str | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscriminationResult:
    status: str
    reason: str
    extracted_answer: str | None
    reference_answer: str | None
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failed(kind: str, raw: str, error: str, *, status: str = "extraction_failed") -> ExtractionResult:
    return ExtractionResult(kind, raw, None, status, error)


def extract_agent_answer(
    raw_response: str,
    *,
    format_kind: str,
    tool_calls: Any = None,
) -> ExtractionResult:
    """Extract an answer only when the requested wire format is explicit."""

    raw = str(raw_response or "")
    kind = str(format_kind).strip().lower()

    if kind in {"function_calling", "tool_call", "tool_calls"}:
        if not isinstance(tool_calls, list) or not tool_calls:
            return _failed(kind, raw, "model returned text without an OpenAI tool_calls object", status="format_invalid")
        for item in tool_calls:
            function = item.get("function") if isinstance(item, dict) else None
            if (
                not isinstance(item, dict)
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function.get("name")
                or "arguments" not in function
            ):
                return _failed(kind, raw, "OpenAI tool_calls contain an invalid function object", status="format_invalid")
        return ExtractionResult(
            kind,
            raw,
            json.dumps(tool_calls, ensure_ascii=False, sort_keys=True),
            "ok",
        )

    if kind == "swe_bench_backticks":
        blocks = _SWE_BENCH_BLOCK.findall(raw)
        if len(blocks) != 1:
            return _failed(
                kind,
                raw,
                f"expected exactly one mswea_bash_command block, found {len(blocks)}",
                status="format_invalid",
            )
        command = blocks[0].strip()
        if not command:
            return _failed(kind, raw, "mswea_bash_command block is empty", status="format_invalid")
        return ExtractionResult(kind, raw, command, "ok")

    if kind in {"choice", "multiple_choice", "multichoice"}:
        answer = extract_choice_answer(raw)
        return (
            ExtractionResult(kind, raw, answer, "ok")
            if answer
            else _failed(kind, raw, "no explicit choice answer was found")
        )

    if kind in {"numeric", "number", "math"}:
        answer = extract_math_answer(raw)
        if not answer:
            return _failed(kind, raw, "no complete numeric answer was found")
        return ExtractionResult(kind, raw, answer, "ok")

    if kind in {"short_answer", "short", "browsecomp"}:
        match = _ANSWER_LINE.search(raw)
        if not match:
            return _failed(kind, raw, "missing explicit Final Answer or Exact Answer line")
        answer = match.group(1).strip()
        return (
            ExtractionResult(kind, raw, answer, "ok")
            if answer
            else _failed(kind, raw, "answer marker is empty")
        )

    if kind in {"code", "code_block", "structured_code"}:
        match = _CODE_BLOCK.fullmatch(raw.strip())
        if not match or not match.group(1).strip():
            return _failed(kind, raw, "expected exactly one non-empty fenced code block", status="format_invalid")
        return ExtractionResult(kind, raw, match.group(1), "ok")

    if kind in {"structured", "json"}:
        candidate = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", candidate, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as error:
            return _failed(kind, raw, f"invalid JSON answer: {error.msg}", status="format_invalid")
        if not isinstance(value, (dict, list)):
            return _failed(kind, raw, "structured answer must be a JSON object or array", status="format_invalid")
        return ExtractionResult(kind, raw, json.dumps(value, ensure_ascii=False, sort_keys=True), "ok")

    return _failed(kind, raw, f"unsupported answer format: {format_kind}", status="format_invalid")


def _same_answer(left: str, right: str, kind: str) -> bool:
    if kind in {"numeric", "number", "math"}:
        try:
            return Decimal(left.strip("$").replace(",", "")) == Decimal(right.strip("$").replace(",", ""))
        except InvalidOperation:
            return False
    if kind in {"structured", "json"}:
        try:
            return json.loads(left) == json.loads(right)
        except json.JSONDecodeError:
            return False
    if kind in {"short_answer", "short", "browsecomp"}:
        return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()
    return left.strip() == right.strip()


def discriminate_agent_result(
    extraction: ExtractionResult,
    *,
    reference_answer: str | None = None,
    transport_status: int | None = 200,
    finish_reason: str | None = None,
) -> DiscriminationResult:
    """Classify transport, format, extraction, and model outcomes separately."""

    if transport_status is not None and transport_status >= 400:
        return DiscriminationResult(
            "interface_error",
            f"upstream HTTP status {transport_status}",
            extraction.extracted_answer,
            reference_answer,
            extraction.raw_response,
        )
    if finish_reason == "length":
        return DiscriminationResult(
            "context_truncated",
            "model completion ended at the context or generation limit",
            extraction.extracted_answer,
            reference_answer,
            extraction.raw_response,
        )
    if extraction.status != "ok":
        return DiscriminationResult(
            extraction.status,
            extraction.error or "answer extraction failed",
            None,
            reference_answer,
            extraction.raw_response,
        )
    if reference_answer is None:
        return DiscriminationResult(
            "unscored",
            "no reference answer was supplied to the diagnostic layer",
            extraction.extracted_answer,
            None,
            extraction.raw_response,
        )
    correct = _same_answer(extraction.extracted_answer or "", str(reference_answer), extraction.format_kind)
    return DiscriminationResult(
        "correct" if correct else "model_error",
        "strict canonical comparison against the supplied reference",
        extraction.extracted_answer,
        str(reference_answer),
        extraction.raw_response,
    )


def write_trace_report(trace_path: Path, output_path: Path, *, exit_code: int | None = None) -> None:
    """Summarize raw proxy traces without replacing the official report."""

    records: list[dict[str, Any]] = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    classified: list[dict[str, Any]] = []
    for record in records:
        request = record.get("request", {})
        source = request.get("json", {}) if isinstance(request, dict) else {}
        response = record.get("response", {})
        body = response.get("body", {}) if isinstance(response, dict) else {}
        choice = {}
        if isinstance(body, dict) and body.get("choices"):
            choice = body["choices"][0] if isinstance(body["choices"][0], dict) else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        kind = "function_calling" if isinstance(source, dict) and source.get("tools") else "short_answer"
        extraction = extract_agent_answer(
            str(content or ""),
            format_kind=kind,
            tool_calls=message.get("tool_calls") if isinstance(message, dict) else None,
        )
        decision = discriminate_agent_result(
            extraction,
            transport_status=response.get("status") if isinstance(response, dict) else None,
            finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        )
        classified.append({
            "path": record.get("path"),
            "decision": decision.to_dict(),
            "extraction": extraction.to_dict(),
        })

    counts: dict[str, int] = {}
    for row in classified:
        status = row["decision"]["status"]
        counts[status] = counts.get(status, 0) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"trace_path": str(trace_path), "exit_code": exit_code, "records": len(records), "counts": counts, "items": classified},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _dataset_format(dataset: str, *, prediction: dict[str, Any], review: dict[str, Any]) -> str:
    name = dataset.casefold()
    metadata = prediction.get("metadata")
    if not isinstance(metadata, dict):
        sample_score = review.get("sample_score")
        metadata = sample_score.get("sample_metadata", {}) if isinstance(sample_score, dict) else {}
    if any(token in name for token in ("general_fc", "bfcl", "function_call", "tau", "tool")):
        return "function_calling"
    if "swe" in name:
        return "swe_bench_backticks"
    if any(token in name for token in ("code", "terminal")):
        return "code"
    if any(token in name for token in ("math", "numeric")):
        return "numeric"
    if any(token in name for token in ("gaia", "browsecomp", "short_answer")):
        return "short_answer"
    if isinstance(metadata, dict) and metadata.get("tools"):
        return "function_calling"
    return "short_answer"


def _model_output_parts(prediction: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None, str | None]:
    output = prediction.get("model_output")
    if not isinstance(output, dict):
        return "", None, None
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", None, output.get("error") if isinstance(output.get("error"), str) else None
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return "", None, str(choice.get("finish_reason") or output.get("error") or "missing message")
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    finish_reason = choice.get("finish_reason") or output.get("stop_reason")
    return str(content or ""), tool_calls if isinstance(tool_calls, list) else None, str(finish_reason) if finish_reason else None


def _official_reports(output_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    report_root = output_dir / "reports"
    if not report_root.is_dir():
        return reports
    for path in sorted(report_root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            reports.append({"path": str(path), "report": value})
    return reports


def _dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    known_prefixes = (
        "automation_bench",
        "bfcl",
        "browsecomp",
        "general_fc",
        "gaia",
        "swe_bench",
        "tau_bench",
        "terminal_bench",
    )
    for prefix in known_prefixes:
        if stem == prefix or stem.startswith(prefix + "_"):
            return prefix
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def write_acceptance_report(output_dir: Path, *, exit_code: int | None = None) -> Path:
    """Join EvalScope official scores with raw model and strict local diagnostics."""

    prediction_root = output_dir / "predictions"
    review_root = output_dir / "reviews"
    samples: list[dict[str, Any]] = []
    for prediction_path in sorted(prediction_root.rglob("*.jsonl")) if prediction_root.is_dir() else []:
        prediction_rows = _read_jsonl(prediction_path)
        review_path = review_root / prediction_path.relative_to(prediction_root)
        review_rows = _read_jsonl(review_path)
        for position, prediction in enumerate(prediction_rows):
            review = review_rows[position] if position < len(review_rows) else {}
            dataset = _dataset_name_from_path(prediction_path)
            format_kind = _dataset_format(dataset, prediction=prediction, review=review)
            raw, tool_calls, finish_reason = _model_output_parts(prediction)
            extraction = extract_agent_answer(raw, format_kind=format_kind, tool_calls=tool_calls)
            reference = review.get("target")
            reference_answer = str(reference) if reference not in (None, "") else None
            transport_status = 200
            output = prediction.get("model_output")
            if isinstance(output, dict) and output.get("error"):
                transport_status = 500
            decision = discriminate_agent_result(
                extraction,
                reference_answer=reference_answer,
                transport_status=transport_status,
                finish_reason=finish_reason,
            )
            sample_score = review.get("sample_score")
            samples.append({
                "dataset": dataset,
                "index": prediction.get("index", position),
                "prediction_path": str(prediction_path),
                "review_path": str(review_path) if review_path.is_file() else None,
                "format_kind": format_kind,
                "reference_answer": reference_answer,
                "official_sample_score": sample_score,
                "extraction": extraction.to_dict(),
                "decision": decision.to_dict(),
                "agent_trace": prediction.get("agent_trace"),
            })

    counts: dict[str, int] = {}
    for sample in samples:
        status = sample["decision"]["status"]
        counts[status] = counts.get(status, 0) + 1
    trace_report_path = output_dir / "raw" / "trace_report.json"
    trace_report: dict[str, Any] | None = None
    if trace_report_path.is_file():
        try:
            value = json.loads(trace_report_path.read_text(encoding="utf-8"))
            trace_report = value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            trace_report = None
    output_path = output_dir / "raw" / "acceptance_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "exit_code": exit_code,
                "counts": counts,
                "samples": samples,
                "official_reports": _official_reports(output_dir),
                "trace_report": trace_report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "DiscriminationResult",
    "ExtractionResult",
    "discriminate_agent_result",
    "extract_agent_answer",
    "write_acceptance_report",
    "write_trace_report",
]
