from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Sequence


_BINARY_METRICS = (
    "acc",
    "acc_norm",
    "exact_match",
    "exact_match_flex",
    "inst_level_strict_acc",
    "prompt_level_strict_acc",
    "pass_at_1",
    "retrieval_score",
)
_PARAGRAPH = re.compile(r"\bparagraph\s+(\d+)\b", re.IGNORECASE)


def build_task_records(
    task_name: str, raw_rows: object
) -> list[dict[str, object]]:
    if not isinstance(raw_rows, list):
        raise ValueError(f"samples for {task_name} must be an array")
    records: list[dict[str, object]] = []
    for sample_index, (row, filter_results) in enumerate(
        _sample_views(task_name, raw_rows)
    ):
        outcome = _binary_outcome_detail(row)
        if outcome is not None:
            metric_name, metric_value = outcome
            if metric_value > 0:
                record = _sample_record(
                    task_name, sample_index, row, judgement_metric=metric_name
                )
                record.update(
                    {
                        "status": "correct",
                        "judgement_metric": metric_name,
                        "judgement_value": metric_value,
                    }
                )
            else:
                record = _bad_case(
                    task_name, sample_index, row, judgement_metric=metric_name
                )
                record["status"] = "incorrect"
                record["judgement_metric"] = metric_name
                record["judgement_value"] = metric_value
        else:
            diagnostic = _quality_diagnostic(task_name, sample_index, row)
            if diagnostic is not None:
                record = diagnostic[1]
                record["status"] = "quality_outlier"
            else:
                record = _sample_record(task_name, sample_index, row)
                record.update(
                    {
                        "status": "unscored",
                        "note": (
                            "No supported binary correctness metric is present; "
                            "this sample is not counted as wrong."
                        ),
                    }
                )
        record["binary_metrics"] = _binary_metrics(row)
        record["model_output"] = _raw_response_text(row)
        selected_filter = row.get("filter")
        if isinstance(selected_filter, str):
            record["filter"] = selected_filter
        if filter_results:
            record["filter_results"] = filter_results
        records.append(record)
    return records


def _sample_views(
    task_name: str, raw_rows: Sequence[object]
) -> list[tuple[Mapping[str, object], dict[str, object]]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[object, ...]] = []
    for row_index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping):
            raise ValueError(f"samples for {task_name} must contain objects")
        filter_name = value.get("filter")
        if isinstance(filter_name, str) and filter_name:
            key = (
                "filtered-doc",
                value.get("doc_id", row_index),
                value.get("doc_hash"),
                value.get("prompt_hash"),
                value.get("target_hash"),
            )
        else:
            key = ("row", row_index)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(value)

    views: list[tuple[Mapping[str, object], dict[str, object]]] = []
    for key in order:
        variants = groups[key]
        selected = min(variants, key=_filter_priority)
        filter_results: dict[str, object] = {}
        if len(variants) > 1:
            for variant in variants:
                filter_name = variant.get("filter")
                if not isinstance(filter_name, str) or not filter_name:
                    continue
                filter_results[filter_name] = {
                    "binary_metrics": _binary_metrics(variant),
                    "filtered_response": _response_text(variant),
                }
        views.append((selected, filter_results))
    return views


def _filter_priority(row: Mapping[str, object]) -> tuple[int, str]:
    name = row.get("filter")
    normalized = name.casefold() if isinstance(name, str) else ""
    priorities = {
        "flexible-extract": 0,
        "none": 1,
        "default": 1,
        "strict-match": 2,
    }
    return priorities.get(normalized, 10), normalized


def analyze_samples(
    samples_by_task: Mapping[str, object],
    *,
    examples_per_task: int = 5,
) -> tuple[dict[str, object], dict[str, object]]:
    if examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive")

    task_summaries: list[dict[str, object]] = []
    selected_cases: list[dict[str, object]] = []
    family_totals: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()

    for task_name, raw_rows in sorted(samples_by_task.items()):
        if not isinstance(task_name, str):
            raise ValueError("samples must map task names to arrays")
        records = build_task_records(task_name, raw_rows)

        errors: list[dict[str, object]] = []
        diagnostics: list[tuple[float, dict[str, object]]] = []
        task_counts: Counter[str] = Counter(samples=len(records))
        error_types: Counter[str] = Counter()
        diagnostic_types: Counter[str] = Counter()
        for record in records:
            status = record["status"]
            if status in {"unscored", "quality_outlier"}:
                task_counts["unscored"] += 1
                if status == "quality_outlier":
                    similarity = record.get("character_similarity")
                    rank = (
                        float(similarity)
                        if isinstance(similarity, (int, float))
                        else 1.0
                    )
                    diagnostics.append((rank, record))
                    diagnostic_types[str(record["error_type"])] += 1
                continue
            task_counts["scored"] += 1
            if status == "correct":
                task_counts["correct"] += 1
                continue
            task_counts["incorrect"] += 1
            error_type = str(record["error_type"])
            error_types[error_type] += 1
            errors.append(record)

        family = _task_family(task_name)
        family_counts = family_totals.setdefault(family, Counter())
        family_counts.update(task_counts)
        totals.update(task_counts)
        task_summaries.append(
            {
                "task_name": task_name,
                "task_family": family,
                **_counts(task_counts),
                "incorrect_rate": _rate(
                    task_counts["incorrect"], task_counts["scored"]
                ),
                "error_types": dict(sorted(error_types.items())),
                "diagnostic_types": dict(sorted(diagnostic_types.items())),
            }
        )
        selected_cases.extend(_evenly_spaced(errors, examples_per_task))
        selected_cases.extend(
            item[1]
            for item in sorted(
                diagnostics,
                key=lambda item: (item[0], str(item[1]["doc_id"])),
            )[:examples_per_task]
        )

    families = [
        {
            "task_family": family,
            **_counts(counts),
            "incorrect_rate": _rate(counts["incorrect"], counts["scored"]),
        }
        for family, counts in sorted(family_totals.items())
    ]
    analysis = {
        "schema_version": 1,
        **_counts(totals),
        "incorrect_rate": _rate(totals["incorrect"], totals["scored"]),
        "task_families": families,
        "tasks": task_summaries,
        "interpretation": {
            "incorrect": (
                "A logged binary correctness metric is zero. Continuous generation "
                "metrics are not converted into pass/fail labels."
            ),
            "unscored": (
                "No supported binary correctness metric is present. These samples "
                "remain available as diagnostics and must not be counted as wrong."
            ),
            "likelihood_tasks": (
                "For loglikelihood tasks, model_answer can be null because the "
                "protocol scores a supplied continuation instead of generating text."
            ),
        },
    }
    bad_cases = {
        "schema_version": 1,
        "selection": {
            "method": "evenly_spaced_by_error_order",
            "max_incorrect_examples_per_task": examples_per_task,
            "max_quality_outliers_per_unscored_task": examples_per_task,
        },
        "cases": selected_cases,
    }
    return analysis, bad_cases


def render_markdown(
    analysis: Mapping[str, object], bad_cases: Mapping[str, object]
) -> str:
    lines = [
        "# lm-eval Error Analysis",
        "",
        "This report separates binary wrong answers from quality diagnostics. "
        "Continuous generation metrics are not treated as pass/fail labels.",
        "",
        "## Coverage",
        "",
        "| Samples | Scored | Incorrect | Error rate | Unscored |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {analysis.get('samples', 0)} | {analysis.get('scored', 0)} | "
            f"{analysis.get('incorrect', 0)} | "
            f"{_percent(analysis.get('incorrect_rate'))} | "
            f"{analysis.get('unscored', 0)} |"
        ),
        "",
        "## Task Families",
        "",
        "| Family | Samples | Scored | Incorrect | Error rate | Unscored |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    families = analysis.get("task_families", [])
    if isinstance(families, list):
        for family in sorted(
            families,
            key=lambda item: (
                -float(item.get("incorrect_rate") or -1),
                str(item.get("task_family", "")),
            ),
        ):
            lines.append(
                f"| {family.get('task_family')} | {family.get('samples', 0)} | "
                f"{family.get('scored', 0)} | {family.get('incorrect', 0)} | "
                f"{_percent(family.get('incorrect_rate'))} | "
                f"{family.get('unscored', 0)} |"
            )

    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
        ]
    )
    cases = bad_cases.get("cases", [])
    displayed_cases = _representative_cases(cases, per_family=3)
    for case in displayed_cases:
        lines.extend(_case_markdown(case))
    if not displayed_cases:
        lines.append("No binary bad cases or generation quality outliers were found.")
    return "\n".join(lines) + "\n"


def render_task_markdown(
    task_summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    displayed_errors: int = 20,
) -> str:
    task_name = str(task_summary.get("task_name", "unknown"))
    errors = [
        record
        for record in records
        if record.get("status") in {"incorrect", "quality_outlier"}
    ]
    lines = [
        f"# {task_name} 评测报告",
        "",
        "`records.jsonl` 包含全部样本；`errors.jsonl` 包含全部错题和生成质量异常。",
        "",
        "## 汇总",
        "",
        "| 样本 | 可判定 | 正确 | 错误 | 错误率 | 不作二元判定 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {task_summary.get('samples', 0)} | "
            f"{task_summary.get('scored', 0)} | "
            f"{task_summary.get('correct', 0)} | "
            f"{task_summary.get('incorrect', 0)} | "
            f"{_percent(task_summary.get('incorrect_rate'))} | "
            f"{task_summary.get('unscored', 0)} |"
        ),
        "",
        "## 错题与质量异常",
        "",
    ]
    for record in errors[:displayed_errors]:
        lines.extend(_case_markdown(record))
    if not errors:
        lines.append("没有发现错题或生成质量异常。")
    elif len(errors) > displayed_errors:
        lines.append(
            f"这里展示 {len(errors)} 条中的前 {displayed_errors} 条；"
            "全部记录见 `errors.jsonl`。"
        )
    return "\n".join(lines) + "\n"


def _binary_outcome_detail(
    row: Mapping[str, object],
) -> tuple[str, float] | None:
    for metric in _BINARY_METRICS:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return metric, numeric
        if isinstance(value, bool):
            return metric, float(value)
    return None


def _binary_metrics(row: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in _BINARY_METRICS:
        value = row.get(metric)
        if isinstance(value, bool):
            metrics[metric] = float(value)
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                metrics[metric] = numeric
    return metrics


def _sample_record(
    task_name: str,
    sample_index: int,
    row: Mapping[str, object],
    judgement_metric: str | None = None,
) -> dict[str, object]:
    detail = _choice_detail(row, judgement_metric)
    if detail is None:
        detail = {
            "model_answer": _response_text(row),
            "standard_answer": _reference_text(row.get("target")),
        }
    return {
        "task_name": task_name,
        "task_family": _task_family(task_name),
        "sample_index": sample_index,
        "doc_id": row.get("doc_id", sample_index),
        "question": _question(row),
        "prompt_excerpt": _prompt_excerpt(row),
        "prompt_diagnostics": _prompt_diagnostics(row),
        "task_metadata": _task_metadata(row),
        **detail,
    }


def _bad_case(
    task_name: str,
    sample_index: int,
    row: Mapping[str, object],
    judgement_metric: str | None = None,
) -> dict[str, object]:
    detail = _choice_detail(row, judgement_metric)
    if detail is not None:
        error_type = (
            "grammar_preference_reversal"
            if task_name.startswith("blimp_")
            else "wrong_choice"
        )
    else:
        response = _response_text(row)
        reference = _reference_text(row.get("target"))
        if task_name.startswith("longbench_passage_retrieval"):
            predicted_paragraph = _paragraph_number(response)
            expected_paragraph = _paragraph_number(reference)
            error_type = (
                "retrieval_format_error"
                if predicted_paragraph is None
                else "wrong_paragraph"
            )
            detail = {
                "model_answer": response,
                "standard_answer": reference,
                "predicted_paragraph": predicted_paragraph,
                "expected_paragraph": expected_paragraph,
            }
        elif task_name.startswith("lambada"):
            error_type = "target_not_greedy"
            detail = {
                "model_answer": None,
                "standard_answer": reference,
                "scored_target_loglikelihood": _first_number(
                    row.get("filtered_resps")
                ),
                "note": (
                    "The task scores the target continuation; it does not "
                    "generate an alternative answer."
                ),
            }
        elif response is None or not response.strip():
            error_type = "empty_response"
            detail = {"model_answer": response, "standard_answer": reference}
        else:
            error_type = "exact_match_failure"
            detail = {"model_answer": response, "standard_answer": reference}

    return {
        "kind": "incorrect",
        "task_name": task_name,
        "task_family": _task_family(task_name),
        "sample_index": sample_index,
        "doc_id": row.get("doc_id", sample_index),
        "error_type": error_type,
        "question": _question(row),
        "prompt_excerpt": _prompt_excerpt(row),
        "prompt_diagnostics": _prompt_diagnostics(row),
        "task_metadata": _task_metadata(row),
        "why_wrong": _why_wrong(error_type, detail),
        **detail,
    }


def _quality_diagnostic(
    task_name: str, sample_index: int, row: Mapping[str, object]
) -> tuple[float, dict[str, object]] | None:
    response = _response_text(row)
    reference = _reference_text(row.get("target"))
    if response is None or reference is None:
        return None
    similarity = SequenceMatcher(
        None, reference.casefold(), response.casefold()
    ).ratio()
    if _has_repetition_loop(response):
        error_type = "repetition_loop"
    elif task_name.startswith("wmt14") and any(
        marker in response.casefold()
        for marker in ("translates to", "the french phrase", "in english")
    ):
        error_type = "translation_meta_answer"
    elif similarity < 0.1:
        error_type = "low_reference_overlap"
    else:
        return None
    return similarity, {
        "kind": "quality_outlier",
        "task_name": task_name,
        "task_family": _task_family(task_name),
        "sample_index": sample_index,
        "doc_id": row.get("doc_id", sample_index),
        "error_type": error_type,
        "question": _question(row),
        "prompt_excerpt": _prompt_excerpt(row),
        "prompt_diagnostics": _prompt_diagnostics(row),
        "task_metadata": _task_metadata(row),
        "model_answer": response,
        "standard_answer": reference,
        "character_similarity": similarity,
        "note": (
            "Diagnostic only: generation quality outliers are not counted as "
            "binary wrong answers."
        ),
        "why_flagged": _why_wrong(error_type, {}),
    }


def _why_wrong(error_type: str, detail: Mapping[str, object]) -> str:
    if error_type in {"wrong_choice", "grammar_preference_reversal"}:
        margin = detail.get("score_margin_over_target")
        suffix = (
            f"；相对标准答案的分数优势={float(margin):.6g}"
            if isinstance(margin, (int, float))
            else ""
        )
        return "模型最高分选项不是标准答案" + suffix
    return {
        "wrong_paragraph": "模型生成的段落编号与标准答案不同。",
        "retrieval_format_error": "模型输出中没有可解析的段落编号。",
        "target_not_greedy": "标准续写不是模型的 greedy continuation。",
        "empty_response": "模型返回了空输出。",
        "exact_match_failure": "模型输出与标准答案未精确匹配。",
        "repetition_loop": "模型输出中的四词片段至少重复了五次。",
        "translation_meta_answer": "模型在解释翻译，而不是直接给出译文。",
        "low_reference_overlap": "模型输出与参考答案的字符相似度低于 0.1。",
    }.get(error_type, error_type)


def _choice_detail(
    row: Mapping[str, object], judgement_metric: str | None = None
) -> dict[str, object] | None:
    arguments = row.get("arguments")
    responses = row.get("filtered_resps")
    if (
        not isinstance(arguments, list)
        or not isinstance(responses, list)
        or len(arguments) < 2
        or len(arguments) != len(responses)
    ):
        return None
    target = _choice_target_index(row, arguments)
    if target is None:
        return None
    scores = [_first_number(response) for response in responses]
    if any(score is None for score in scores):
        return None
    raw_scores = [float(score) for score in scores if score is not None]
    choices = [
        _display_choice(row, index, _argument_continuation(argument))
        for index, argument in enumerate(arguments)
    ]
    if judgement_metric == "acc_norm":
        lengths = [
            max(1, len(choice.strip())) if isinstance(choice, str) else 1
            for choice in choices
        ]
        numeric_scores = [
            score / length for score, length in zip(raw_scores, lengths, strict=True)
        ]
        scoring = "character_length_normalized_loglikelihood"
    else:
        numeric_scores = raw_scores
        scoring = "loglikelihood"
    predicted = max(range(len(numeric_scores)), key=numeric_scores.__getitem__)
    return {
        "model_answer": choices[predicted],
        "standard_answer": choices[target],
        "predicted_choice_index": predicted,
        "expected_choice_index": target,
        "choice_scores": numeric_scores,
        "raw_choice_scores": raw_scores,
        "choice_scoring": scoring,
        "score_margin_over_target": numeric_scores[predicted] - numeric_scores[target],
        "choices": choices,
    }


def _choice_target_index(
    row: Mapping[str, object], arguments: Sequence[object]
) -> int | None:
    target = row.get("target")
    if isinstance(target, int) and not isinstance(target, bool):
        return target if 0 <= target < len(arguments) else None
    doc = row.get("doc")
    if isinstance(doc, Mapping):
        answer_index = doc.get("answer_index")
        if isinstance(answer_index, int) and not isinstance(answer_index, bool):
            return answer_index if 0 <= answer_index < len(arguments) else None
    if isinstance(target, str):
        normalized_target = target.strip()
        for index, argument in enumerate(arguments):
            continuation = _argument_continuation(argument)
            if isinstance(continuation, str) and continuation.strip() == normalized_target:
                return index
    return None


def _display_choice(
    row: Mapping[str, object], index: int, continuation: str | None
) -> str | None:
    doc = row.get("doc")
    if not isinstance(doc, Mapping):
        return continuation
    candidate: object = None
    for key in ("choices", "options"):
        values = doc.get(key)
        if isinstance(values, list) and index < len(values):
            candidate = values[index]
            break
    if candidate is None:
        candidate = doc.get(chr(ord("A") + index))
    if not isinstance(candidate, str):
        return continuation
    label = continuation.strip() if isinstance(continuation, str) else ""
    if len(label) == 1 and label.casefold() in "abcdefghijklmnopqrstuvwxyz":
        return f"{label}. {candidate}"
    return candidate


def _first_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, list):
        for item in value:
            found = _first_number(item)
            if found is not None:
                return found
    return None


def _argument_continuation(value: object) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        continuation = value[1]
        return continuation if isinstance(continuation, str) else repr(continuation)
    return None


def _response_text(row: Mapping[str, object]) -> str | None:
    value: object = row.get("filtered_resps")
    while isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, str) else None


def _raw_response_text(row: Mapping[str, object]) -> str | None:
    value: object = row.get("resps")
    while isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, str) else None


def _reference_text(value: object) -> str | None:
    while isinstance(value, list) and value:
        value = value[0]
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _question(row: Mapping[str, object]) -> str | None:
    doc = row.get("doc")
    if isinstance(doc, Mapping):
        for key in ("question", "Question", "query", "sentence", "text"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return _bounded(value, 1600, tail=key == "text")
        translation = doc.get("translation")
        if isinstance(translation, Mapping):
            source = translation.get("en")
            if isinstance(source, str):
                return source
    prompt = _prompt_text(row)
    return _bounded(prompt, 1600, tail=True) if prompt else None


def _prompt_excerpt(row: Mapping[str, object]) -> str | None:
    prompt = _prompt_text(row)
    return _bounded(prompt, 1600, tail=True) if prompt else None


def _prompt_diagnostics(row: Mapping[str, object]) -> dict[str, object]:
    prompt = _prompt_text(row)
    if prompt is None:
        return {}
    repeated_markers = {
        marker: prompt.count(marker)
        for marker in (
            "Here are 30 paragraphs from Wikipedia",
            "The following is an abstract.",
            "Please enter the number of the paragraph",
        )
        if prompt.count(marker) > 1
    }
    return {
        "prompt_characters": len(prompt),
        "repeated_instruction_markers": repeated_markers,
    }


def _prompt_text(row: Mapping[str, object]) -> str | None:
    arguments = row.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        return None
    first = arguments[0]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
        return first[0]
    return None


def _task_metadata(row: Mapping[str, object]) -> dict[str, object]:
    doc = row.get("doc")
    if not isinstance(doc, Mapping):
        return {}
    keys = (
        "UID",
        "field",
        "linguistics_term",
        "dataset",
        "language",
        "length",
        "subject",
    )
    return {key: doc[key] for key in keys if key in doc}


def _task_family(task_name: str) -> str:
    for prefix, family in (
        ("blimp_", "blimp"),
        ("cmmlu_", "cmmlu"),
        ("longbench_", "longbench"),
    ):
        if task_name.startswith(prefix):
            return family
    return task_name


def _paragraph_number(text: str | None) -> int | None:
    if text is None:
        return None
    match = _PARAGRAPH.search(text)
    return int(match.group(1)) if match else None


def _has_repetition_loop(value: str) -> bool:
    words = value.casefold().split()
    if len(words) < 8:
        return False
    four_grams = Counter(
        tuple(words[index : index + 4]) for index in range(len(words) - 3)
    )
    return max(four_grams.values(), default=0) >= 5


def _bounded(value: str, limit: int, *, tail: bool = False) -> str:
    if len(value) <= limit:
        return value
    if tail:
        return "[...]" + value[-limit:]
    return value[:limit] + "[...]"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _counts(values: Counter[str]) -> dict[str, int]:
    return {
        key: values[key]
        for key in ("samples", "scored", "correct", "incorrect", "unscored")
    }


def _evenly_spaced(
    rows: Sequence[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    if len(rows) <= limit:
        return list(rows)
    if limit == 1:
        return [rows[0]]
    indices = [
        round(index * (len(rows) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [rows[index] for index in indices]


def _percent(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _representative_cases(value: object, *, per_family: int) -> list[object]:
    if not isinstance(value, list):
        return []
    counts: Counter[str] = Counter()
    seen_tasks: dict[str, set[str]] = {}
    selected: list[object] = []
    selected_ids: set[int] = set()
    candidates = [case for case in value if isinstance(case, Mapping)]
    for prefer_new_task in (True, False):
        for case in candidates:
            if id(case) in selected_ids:
                continue
            family = str(case.get("task_family", case.get("task_name", "unknown")))
            task_name = str(case.get("task_name", "unknown"))
            family_tasks = seen_tasks.setdefault(family, set())
            if counts[family] >= per_family:
                continue
            if prefer_new_task and task_name in family_tasks:
                continue
            if not prefer_new_task and task_name not in family_tasks:
                continue
            counts[family] += 1
            family_tasks.add(task_name)
            selected.append(case)
            selected_ids.add(id(case))
    return selected


def _case_markdown(case: object) -> list[str]:
    if not isinstance(case, Mapping):
        return []
    title = (
        f"### {case.get('task_name')} / doc_id={case.get('doc_id')} / "
        f"{case.get('error_type')}"
    )
    lines = [title, ""]
    for label, key in (
        ("问题", "question"),
        ("模型原始输出", "model_output"),
        ("模型答案", "model_answer"),
        ("标准答案", "standard_answer"),
        ("判错原因", "why_wrong"),
        ("异常原因", "why_flagged"),
        ("说明", "note"),
    ):
        value = case.get(key)
        if value is not None:
            rendered = str(value).replace("\n", " ").strip()
            if key == "model_output":
                rendered = _bounded(rendered, 2400)
            lines.append(f"- {label}: {rendered}")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helicopter-lm-eval-analysis")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--examples-per-task", type=int, default=5)
    args = parser.parse_args(argv)
    raw = json.loads(args.results.read_text(encoding="utf-8"))
    samples = raw.get("samples")
    if not isinstance(samples, Mapping):
        raise SystemExit("results artifact does not contain logged samples")
    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    from .artifacts import write_posthoc_analysis

    write_posthoc_analysis(
        output_dir=output_dir,
        results_path=args.results,
        samples=samples,
        examples_per_task=args.examples_per_task,
    )
    print(f"lm-eval error analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
