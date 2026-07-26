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


def extract_agent_answer(raw_response: str, *, format_kind: str) -> ExtractionResult:
    """Extract an answer only when the requested wire format is explicit."""

    raw = str(raw_response or "")
    kind = str(format_kind).strip().lower()

    if kind in {"function_calling", "tool_call", "tool_calls"}:
        return _failed(kind, raw, "model returned text without an OpenAI tool_calls object", status="format_invalid")

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
        extraction = extract_agent_answer(str(content or ""), format_kind=kind)
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


__all__ = [
    "DiscriminationResult",
    "ExtractionResult",
    "discriminate_agent_result",
    "extract_agent_answer",
    "write_trace_report",
]
