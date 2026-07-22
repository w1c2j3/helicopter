from __future__ import annotations

from typing import Any

from .charts import serialize_charts
from .normalize import (
    display_param,
    entry_domain,
    is_naive,
    json_key,
    metric_from_context,
    parse_model_tags,
    score_to_percent,
)


def build_leaderboard_payload(
    entries: list[dict[str, Any]],
    *,
    selected_model: str | None,
    view: str,
    tuning_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    is_delta = view.endswith("_delta")
    is_field_avg = view.startswith("field_avg")
    visible = _select_entries(entries, selected_model)
    param_columns = _param_columns(visible, is_delta=is_delta)
    domains = []
    for group in (
        "knowledge",
        "math",
        "coding",
        "agent",
        "instruction_following",
        "function_call",
    ):
        rows = _rows_for_domain(visible, group, param_columns, is_delta=is_delta)
        label = next(item["label"] for item in _domain_groups_with_naive() if item["key"] == group)
        title = next(item["title"] for item in _domain_groups_with_naive() if item["key"] == group)
        domains.append(
            {
                "key": group,
                "title": title,
                "label": label,
                "param_columns": param_columns,
                "rows": rows,
            }
        )
    naive_entries = [entry for entry in entries if is_naive(entry.get("task"), entry.get("sampling_config"))]
    naive_columns = _param_columns(naive_entries, is_delta=is_delta)
    payload = {
        "view": view,
        "view_label": _table_view_label(view),
        "is_delta": is_delta,
        "is_field_avg": is_field_avg,
        "param_columns": param_columns,
        "interaction_meta": {},
        "domains": domains,
        "naive_board": {
            "key": "naive",
            "title": "朴素榜",
            "label": "朴素榜",
            "is_delta": is_delta,
            "param_columns": naive_columns,
            "rows": _rows_for_domain(naive_entries, None, naive_columns, is_delta=is_delta),
        },
        "overview": _overview(domains, param_columns, is_delta=is_delta) if is_field_avg else None,
        "selection": _selection(entries, visible, selected_model),
        "charts": serialize_charts(visible),
        "matrix": _leaderboard_matrix(entries),
        "tuning_matrix": _tuning_matrix(tuning_entries or entries),
        "errors": [],
    }
    for domain in domains:
        for row in domain["rows"]:
            for cell in row["cells"]:
                for key in ("meta", "prev_meta", "latest_meta"):
                    meta = cell.get(key)
                    if meta:
                        payload["interaction_meta"][meta["cell_id"]] = meta
    return payload


def _leaderboard_matrix(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the primary board: Naive-only, model rows and benchmark columns."""
    naive_entries = [entry for entry in entries if _is_primary_entry(entry)]
    domains = [_matrix_domain(naive_entries, group) for group in _domain_groups_with_naive() if group["key"] != "naive"]
    return {
        "model_count": len({str(entry.get("model") or "") for entry in naive_entries if entry.get("model")}),
        "benchmark_count": sum(len(domain["columns"]) for domain in domains),
        "domains": domains,
    }


def _is_primary_entry(entry: dict[str, Any]) -> bool:
    if entry.get("is_param_search"):
        return False
    sampling_config = entry.get("sampling_config")
    # Historical DB rows can predate prompt metadata. They are official
    # non-search runs, so absence of that metadata must not hide them.
    return not sampling_config or is_naive(entry.get("task"), sampling_config)


def _matrix_domain(entries: list[dict[str, Any]], group: dict[str, str]) -> dict[str, Any]:
    domain_entries = [entry for entry in entries if entry_domain(entry) == group["key"]]
    column_entries: dict[str, list[dict[str, Any]]] = {}
    column_data: dict[str, dict[str, Any]] = {}
    for entry in domain_entries:
        metric, _ = metric_from_context(entry.get("metrics") or {}, entry.get("sampling_config"))
        if metric is None:
            continue
        key = _entry_matrix_column_key(entry, metric)
        column_entries.setdefault(key, []).append(entry)
        column_data[key] = {
            "key": key,
            "label": str(entry.get("dataset") or "unknown"),
            "metric": metric,
            "eval_method": str(entry.get("cot_mode") or ("CoT" if entry.get("cot") else "NoCoT")),
            "num_samples": entry.get("samples") or entry.get("problems"),
        }
    columns = [
        column_data[key]
        for key in sorted(
            column_data,
            key=lambda item: (
                column_data[item]["label"].lower(),
                column_data[item]["eval_method"].lower(),
                str(column_data[item]["metric"]).lower(),
            ),
        )
    ]
    models = sorted({str(entry.get("model")) for entry in domain_entries if entry.get("model")})
    rows = []
    for model in models:
        cells = []
        for column in columns:
            candidates = [entry for entry in column_entries[column["key"]] if str(entry.get("model")) == model]
            entry = max(candidates, key=_entry_order, default=None)
            cells.append(_matrix_cell(entry, column))
        values = [cell["percent"] for cell in cells if cell["percent"] is not None]
        rows.append(
            {
                "rank": 0,
                "model": model,
                "param": display_param(parse_model_tags(model)[2]),
                "average": sum(values) / len(values) if values else None,
                "coverage": len(values),
                "cells": cells,
            }
        )
    rows.sort(
        key=lambda row: (
            row["average"] is None,
            -(row["average"] or 0),
            row["model"].lower(),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {**group, "columns": columns, "rows": rows}


def _entry_matrix_column_key(entry: dict[str, Any], metric: str) -> str:
    del metric
    return str(entry.get("dataset") or "unknown")


def _matrix_cell(entry: dict[str, Any] | None, column: dict[str, Any]) -> dict[str, Any]:
    if entry is None:
        return {
            "percent": None,
            "meta": None,
            "metric": column.get("metric"),
            "num_samples": None,
            "created_at": None,
        }
    metric = str(column.get("metric") or "")
    created_at = entry.get("created_at")
    return {
        "percent": _entry_percent(entry, metric),
        "meta": _cell_meta(
            entry,
            str(entry.get("dataset") or column.get("label") or ""),
            str(entry.get("cot_mode") or ("CoT" if entry.get("cot") else "NoCoT")),
            metric,
            entry.get("model"),
        ),
        "metric": metric,
        "num_samples": entry.get("samples") or entry.get("problems"),
        "created_at": created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else (str(created_at) if created_at else None),
    }


def _tuning_matrix(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the Normal/parameter-search board: weights rows and prompt/sampling columns."""
    tuning_entries = [
        entry
        for entry in entries
        if entry.get("is_param_search") or not is_naive(entry.get("task"), entry.get("sampling_config"))
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    benchmark_data: dict[str, dict[str, Any]] = {}
    for entry in tuning_entries:
        metric, _ = metric_from_context(entry.get("metrics") or {}, entry.get("sampling_config"))
        if metric is None:
            continue
        key = json_key([str(entry.get("dataset") or "unknown"), metric])
        grouped.setdefault(key, []).append(entry)
        benchmark_data[key] = {
            "key": key,
            "label": str(entry.get("dataset") or "unknown"),
            "metric": metric,
            "num_samples": entry.get("samples") or entry.get("problems"),
        }
    benchmarks = []
    for benchmark_key in sorted(grouped, key=lambda key: benchmark_data[key]["label"].lower()):
        benchmark_entries = grouped[benchmark_key]
        configs: dict[str, dict[str, Any]] = {}
        by_config: dict[str, list[dict[str, Any]]] = {}
        for entry in benchmark_entries:
            config = _tuning_config(entry)
            configs[config["key"]] = config
            by_config.setdefault(config["key"], []).append(entry)
        columns = [configs[key] for key in sorted(configs, key=lambda key: configs[key]["label"].lower())]
        models = sorted({str(entry.get("model")) for entry in benchmark_entries if entry.get("model")})
        rows = []
        for model in models:
            cells = []
            for column in columns:
                candidates = [entry for entry in by_config[column["key"]] if str(entry.get("model")) == model]
                entry = max(candidates, key=_entry_order, default=None)
                cells.append(_matrix_cell(entry, benchmark_data[benchmark_key]))
            values = [cell["percent"] for cell in cells if cell["percent"] is not None]
            rows.append(
                {
                    "rank": 0,
                    "model": model,
                    "param": display_param(parse_model_tags(model)[2]),
                    "best": max(values) if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "coverage": len(values),
                    "cells": cells,
                }
            )
        rows.sort(
            key=lambda row: (
                row["best"] is None,
                -(row["best"] or 0),
                row["model"].lower(),
            )
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        benchmarks.append({**benchmark_data[benchmark_key], "columns": columns, "rows": rows})
    return {"benchmark_count": len(benchmarks), "benchmarks": benchmarks}


def _tuning_config(entry: dict[str, Any]) -> dict[str, Any]:
    outer = entry.get("sampling_config") if isinstance(entry.get("sampling_config"), dict) else {}
    nested = outer.get("sampling_config") if isinstance(outer.get("sampling_config"), dict) else outer
    answer = nested.get("answer") if isinstance(nested.get("answer"), dict) else nested
    prompt_profile = str(outer.get("prompt_profile") or "normal")
    cot_mode = str(entry.get("cot_mode") or outer.get("cot_mode") or ("CoT" if entry.get("cot") else "NoCoT"))
    parts = [prompt_profile, cot_mode]
    for key, prefix in (
        ("temperature", "T"),
        ("top_k", "K"),
        ("top_p", "P"),
        ("presence_penalty", "PP"),
        ("frequency_penalty", "FP"),
    ):
        if answer.get(key) is not None:
            parts.append(f"{prefix}{answer[key]}")
    config_key = json_key({"profile": prompt_profile, "cot_mode": cot_mode, "sampling": nested})
    return {
        "key": config_key,
        "label": " · ".join(parts),
        "prompt_profile": prompt_profile,
        "cot_mode": cot_mode,
        "sampling_config": nested,
    }


def _entry_order(entry: dict[str, Any]) -> tuple[Any, int]:
    return entry.get("created_at") or "", int(entry.get("score_id") or 0)


def build_meta_payload(entries: list[dict[str, Any]], errors: list[str] | None = None) -> dict[str, Any]:
    models = sorted({str(entry["model"]) for entry in entries})
    return {
        "auto_label": "每档最新（调度策略）",
        "default_view": "benchmark_detail_delta",
        "table_views": [
            {"key": "benchmark_detail_latest", "label": "明细（最新）"},
            {"key": "field_avg_latest", "label": "领域均分（最新）"},
            {"key": "benchmark_detail_delta", "label": "明细（上一代 vs 最新）"},
            {"key": "field_avg_delta", "label": "领域均分（上一代 vs 最新）"},
        ],
        "domain_groups": _domain_groups_with_naive(),
        "models": models,
        "model_choices": ["每档最新（调度策略）", *models],
        "entry_count": len(entries),
        "errors": errors or [],
    }


def _domain_groups_with_naive() -> list[dict[str, str]]:
    return [
        {
            "key": "knowledge",
            "label": "Knowledge",
            "title": "知识类（MMLU / Multi-choice）",
        },
        {"key": "math", "label": "Math", "title": "数学推理（AIME / Math-500 等）"},
        {"key": "coding", "label": "Coding", "title": "代码"},
        {"key": "agent", "label": "Agent", "title": "Agent 工作流"},
        {
            "key": "instruction_following",
            "label": "Instruction Following",
            "title": "指令遵循（IFEval 等）",
        },
        {"key": "function_call", "label": "Function Call", "title": "函数调用"},
        {"key": "naive", "label": "朴素榜", "title": "朴素榜"},
    ]


def _table_view_label(view: str) -> str:
    labels = {
        "benchmark_detail_latest": "明细（最新）",
        "field_avg_latest": "领域均分（最新）",
        "benchmark_detail_delta": "明细（上一代 vs 最新）",
        "field_avg_delta": "领域均分（上一代 vs 最新）",
    }
    return labels.get(view, labels["benchmark_detail_delta"])


def _select_entries(entries: list[dict[str, Any]], selected_model: str | None) -> list[dict[str, Any]]:
    if selected_model and selected_model != "每档最新（调度策略）":
        return [entry for entry in entries if entry["model"] == selected_model]
    return entries


def _param_columns(entries: list[dict[str, Any]], *, is_delta: bool) -> list[dict[str, Any]]:
    by_param: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        _, _, param = parse_model_tags(str(entry["model"]))
        by_param.setdefault(param, []).append(entry)
    columns = []
    for param, param_entries in sorted(by_param.items(), key=lambda item: item[0]):
        models = sorted(
            {str(entry["model"]) for entry in param_entries},
            key=lambda model: max(e["created_at"] for e in param_entries if e["model"] == model),
        )
        latest = models[-1] if models else ""
        prev = models[-2] if is_delta and len(models) > 1 else None
        columns.append(
            {
                "param": param,
                "param_label": display_param(param),
                "latest_model": latest,
                "latest_label": latest,
                "prev_model": prev,
                "prev_label": prev,
            }
        )
    return columns


def _rows_for_domain(
    entries: list[dict[str, Any]],
    domain: str | None,
    param_columns: list[dict[str, Any]],
    *,
    is_delta: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        dataset = str(entry["dataset"])
        if domain is not None and entry_domain(entry) != domain:
            continue
        metric, _ = metric_from_context(entry.get("metrics") or {}, entry.get("sampling_config"))
        if metric is None:
            continue
        eval_method = "cot" if entry.get("cot") else "nocot"
        grouped.setdefault((dataset, eval_method, metric), []).append(entry)
    rows = []
    for (dataset, eval_method, metric), group_entries in sorted(grouped.items()):
        cells = []
        for column in param_columns:
            latest = _entry_for_model(group_entries, column.get("latest_model"))
            prev = _entry_for_model(group_entries, column.get("prev_model"))
            latest_percent = _entry_percent(latest, metric)
            prev_percent = _entry_percent(prev, metric)
            if is_delta:
                delta = None if latest_percent is None or prev_percent is None else latest_percent - prev_percent
                cells.append(
                    {
                        "prev": prev_percent,
                        "latest": latest_percent,
                        "delta": delta,
                        "prev_meta": _cell_meta(prev, dataset, eval_method, metric, column.get("prev_label")),
                        "latest_meta": _cell_meta(
                            latest,
                            dataset,
                            eval_method,
                            metric,
                            column.get("latest_label"),
                        ),
                    }
                )
            else:
                cells.append(
                    {
                        "percent": latest_percent,
                        "meta": _cell_meta(
                            latest,
                            dataset,
                            eval_method,
                            metric,
                            column.get("latest_label"),
                        ),
                    }
                )
        rows.append(
            {
                "benchmark_name": dataset,
                "num_samples": max((entry.get("samples") or 0 for entry in group_entries), default=0) or None,
                "eval_method": eval_method,
                "k_metric": metric,
                "cells": cells,
            }
        )
    return rows


def _entry_for_model(entries: list[dict[str, Any]], model: Any) -> dict[str, Any] | None:
    if not model:
        return None
    candidates = [entry for entry in entries if entry["model"] == model]
    return max(
        candidates,
        key=lambda entry: (entry["created_at"], entry.get("score_id") or 0),
        default=None,
    )


def _entry_percent(entry: dict[str, Any] | None, metric: str) -> float | None:
    if entry is None:
        return None
    _, value = metric_from_context(entry.get("metrics") or {}, entry.get("sampling_config"))
    if metric in (entry.get("metrics") or {}):
        value = (entry.get("metrics") or {}).get(metric)
    try:
        return score_to_percent(float(value))
    except (TypeError, ValueError):
        return None


def _cell_meta(
    entry: dict[str, Any] | None,
    dataset: str,
    eval_method: str,
    metric: str,
    label: Any,
) -> dict[str, Any] | None:
    if entry is None:
        return None
    task_id = entry.get("task_id")
    return {
        "cell_id": f"{task_id}:{dataset}:{metric}:{label}",
        "task_id": task_id,
        "benchmark_name": dataset,
        "eval_method": eval_method,
        "k_metric": metric,
        "column_label": str(label or ""),
        "model": entry.get("model"),
        "tooltip": None,
        "clickable": task_id is not None,
    }


def _overview(domains: list[dict[str, Any]], columns: list[dict[str, Any]], *, is_delta: bool) -> list[dict[str, Any]]:
    rows = []
    for domain in domains:
        cells = []
        for idx, _column in enumerate(columns):
            values: list[float] = []
            latest_values: list[float] = []
            prev_values: list[float] = []
            for row in domain["rows"]:
                cell = row["cells"][idx]
                if is_delta:
                    if cell["latest"] is not None:
                        latest_values.append(cell["latest"])
                    if cell["prev"] is not None:
                        prev_values.append(cell["prev"])
                elif cell["percent"] is not None:
                    values.append(cell["percent"])
            if is_delta:
                latest = sum(latest_values) / len(latest_values) if latest_values else None
                prev = sum(prev_values) / len(prev_values) if prev_values else None
                cells.append(
                    {
                        "prev": prev,
                        "latest": latest,
                        "delta": None if latest is None or prev is None else latest - prev,
                    }
                )
            else:
                cells.append({"percent": sum(values) / len(values) if values else None})
        rows.append(
            {
                "domain_key": domain["key"],
                "domain_title": domain["title"],
                "cells": cells,
            }
        )
    return rows


def _selection(
    entries: list[dict[str, Any]],
    visible: list[dict[str, Any]],
    selected_model: str | None,
) -> dict[str, Any]:
    value = selected_model or "每档最新（调度策略）"
    return {
        "dropdown_value": value,
        "selected_label": value,
        "auto_selected": value == "每档最新（调度策略）",
        "model_sequence": sorted({entry["model"] for entry in visible}),
        "skipped_small_params": max(
            0,
            len({entry["model"] for entry in entries}) - len({entry["model"] for entry in visible}),
        ),
        "auto_label": "每档最新（调度策略）",
    }
