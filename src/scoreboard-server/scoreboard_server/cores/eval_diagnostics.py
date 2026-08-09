from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


TRUNCATION_STOP_REASONS = frozenset({"length", "max_length", "max_tokens"})
_SIMPLE_CHOICE = re.compile(r"^\s*[\[(]?[A-Z][\])]?\s*$")


def final_generation_diagnostics(
    context: Mapping[str, Any] | None,
    *,
    domain: str,
) -> dict[str, Any]:
    """Describe the evaluator-facing generation, never an earlier thinking stage.

    The precedence and truncation semantics mirror
    ``ops/g1i_strict46/report_final_truncation_matrix.py``:

    * Math observes only stage 2 (``stages[1]``); stage 1 is reasoning and is
      intentionally excluded.
    * Knowledge observes the raw answer bridge selected by the evaluator.
    * Coding and instruction-following observe their direct/single generation.

    Missing telemetry is explicit and is never interpreted as a clean stop.
    """

    payload = context if isinstance(context, Mapping) else {}
    normalized_domain = str(domain or "").strip().lower()
    if normalized_domain == "math":
        return _math_final_diagnostics(payload)
    if normalized_domain == "knowledge":
        return _knowledge_final_diagnostics(payload)
    return _ordinary_final_diagnostics(payload)


def _math_final_diagnostics(context: Mapping[str, Any]) -> dict[str, Any]:
    stages = context.get("stages")
    # Strict math has two generations: stages[0] is reasoning and stages[1]
    # is the answer submitted to the evaluator.  Never fall back to stage 0.
    if not isinstance(stages, list) or len(stages) < 2 or not isinstance(stages[1], Mapping):
        return _result(None, observed=False, truncated=False)

    final_stage = stages[1]
    reason, _reason_present = _field(final_stage, "stop_reason")
    stats = context.get("stats")
    stage2_stats = stats.get("stage2") if isinstance(stats, Mapping) else None
    stats_truncated = (
        _truthy(stage2_stats.get("truncated"))
        if isinstance(stage2_stats, Mapping) and "truncated" in stage2_stats
        else False
    )
    return _result(
        reason,
        # The persisted final stage itself is observable even when an old
        # producer omitted its stop_reason field, matching the audit query.
        observed=True,
        truncated=stats_truncated or _is_truncation_reason(reason),
    )


def _knowledge_final_diagnostics(context: Mapping[str, Any]) -> dict[str, Any]:
    bridges = context.get("format_bridges")
    if isinstance(bridges, Mapping):
        for reason_key, completion_key in (
            ("answer_stage_raw_stop_reason", "answer_stage_raw_completion"),
            ("strategy_b_final_raw_stop_reason", "strategy_b_final_raw_completion"),
        ):
            reason, present = _field(bridges, reason_key)
            if present:
                completion = str(bridges.get(completion_key) or "")
                return _result(
                    reason,
                    observed=True,
                    truncated=(
                        _is_truncation_reason(reason)
                        and _SIMPLE_CHOICE.fullmatch(completion) is None
                    ),
                )

    reason, present = _field(context, "direct_raw_finish_reason")
    if present:
        completion = str(context.get("direct_raw_completion") or "")
        return _result(
            reason,
            observed=True,
            truncated=(
                _is_truncation_reason(reason)
                and _SIMPLE_CHOICE.fullmatch(completion) is None
            ),
        )

    reason, present = _field(context, "strategy_a_stop_reason")
    if present:
        return _result(
            reason,
            observed=True,
            truncated=_is_truncation_reason(reason),
        )

    strategy_a = context.get("strategy_a")
    if isinstance(strategy_a, Mapping):
        reason, present = _field(strategy_a, "stop_reason")
        if present:
            return _result(
                reason,
                observed=True,
                truncated=_is_truncation_reason(reason),
            )

    return _result(None, observed=False, truncated=False)


def _ordinary_final_diagnostics(context: Mapping[str, Any]) -> dict[str, Any]:
    reason, present = _field(context, "direct_raw_finish_reason")
    if present:
        return _result(
            reason,
            observed=True,
            truncated=_is_truncation_reason(reason),
        )

    stages = context.get("stages")
    if isinstance(stages, list) and stages and isinstance(stages[0], Mapping):
        reason, reason_present = _field(stages[0], "stop_reason")
        if reason_present:
            return _result(
                reason,
                observed=True,
                truncated=_is_truncation_reason(reason),
            )

    stats = context.get("stats")
    if isinstance(stats, Mapping) and "truncated" in stats:
        truncated = _truthy(stats.get("truncated"))
        return _result(
            "stats.truncated" if truncated else None,
            observed=True,
            truncated=truncated,
        )

    return _result(None, observed=False, truncated=False)


def _field(mapping: Mapping[str, Any], key: str) -> tuple[str | None, bool]:
    if key not in mapping:
        return None, False
    value = mapping.get(key)
    if value is None:
        return None, True
    text = str(value).strip()
    return text or None, True


def _is_truncation_reason(reason: str | None) -> bool:
    return str(reason or "").strip().lower() in TRUNCATION_STOP_REASONS


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _result(
    reason: str | None,
    *,
    observed: bool,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "final_stop_reason": reason,
        "final_stop_telemetry_observed": observed,
        "is_truncated": truncated,
    }
