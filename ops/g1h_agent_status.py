#!/usr/bin/env python3
"""Read-only status report for the formal g1h EvalScope Agent matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


CATALOG = Path(
    "/home/rwkv/chase/helicopter-e0eeddc/benchmarks/evalscope_agent_datasets.json"
)
MODELS = {
    "1.5B": "%g1h-1.5b%",
    "2.9B": "%g1h-2.9b%",
    "7.2B": "%g1h-7.2b%",
    "13.3B": "%g1h-13.3b%",
}


def metric_score(raw: object) -> object:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):
        return raw.get("score")
    return None


AGGREGATE_ONLY_SAMPLES = {
    "k2_verifier": 2000,
    "minimax_verifier": 102,
}


def context_audit(raw: object) -> dict[str, object]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict) and isinstance(raw.get("context_audit"), dict):
        return raw["context_audit"]
    return {}


def run_psql(query: str) -> str:
    command = [
        "/usr/bin/psql",
        "-XAt",
        "-v", "ON_ERROR_STOP=1",
        "-h", os.environ.get("SCOREBOARD_DB_HOST", "127.0.0.1"),
        "-p", os.environ.get("SCOREBOARD_DB_PORT", "55433"),
        "-U", os.environ["SCOREBOARD_DB_USER"],
        "-d", os.environ["SCOREBOARD_DB_NAME"],
        "-c", query,
    ]
    process_env = os.environ.copy()
    process_env["PGPASSWORD"] = os.environ["SCOREBOARD_DB_PASSWORD"]
    completed = subprocess.run(
        command, env=process_env, check=False, text=True, capture_output=True
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or f"psql exited {completed.returncode}")
    return completed.stdout


def main() -> None:
    if "--schema" in sys.argv:
        print(
            run_psql(
                "SELECT table_name || '|' || column_name "
                "FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name IN "
                "('task','model','benchmark','scores','completions','eval') "
                "ORDER BY table_name,ordinal_position"
            ),
            end="",
        )
        return
    datasets = [item["name"] for item in json.loads(CATALOG.read_text())["datasets"]]
    dataset_sql = ",".join("'" + name.replace("'", "''") + "'" for name in datasets)
    pattern_sql = " OR ".join(
        "m.name LIKE '" + pattern.replace("'", "''") + "'" for pattern in MODELS.values()
    )
    query = f"""
      SELECT row_to_json(output_row) FROM (
        WITH ranked AS (
            SELECT
                s.score_id,
                s.task_id,
                m.model_name,
                b.benchmark_name AS benchmark,
                s.metrics,
                row_number() OVER (
                    PARTITION BY m.model_name, b.benchmark_name ORDER BY s.score_id DESC
                ) AS rank
            FROM scores AS s
            JOIN task AS t ON t.task_id = s.task_id
            JOIN model AS m ON m.model_id = t.model_id
            JOIN benchmark AS b ON b.benchmark_id = t.benchmark_id
            WHERE b.benchmark_name IN ({dataset_sql})
              AND ({pattern_sql.replace('m.name', 'm.model_name')})
        )
        SELECT
            r.score_id,
            r.task_id,
            r.model_name,
            r.benchmark,
            r.metrics,
            (SELECT count(*) FROM completions AS c WHERE c.task_id = r.task_id) AS completions,
            (
                SELECT count(*)
                FROM eval AS e
                JOIN completions AS ec ON ec.completions_id = e.completions_id
                WHERE ec.task_id = r.task_id
            ) AS evals
        FROM ranked AS r
        WHERE r.rank = 1
        ORDER BY r.score_id
      ) AS output_row
    """
    rows = [json.loads(line) for line in run_psql(query).splitlines() if line.strip()]

    latest = []
    completed_pairs: set[tuple[str, str]] = set()
    validated_pairs: set[tuple[str, str]] = set()
    for row in rows:
        score_id = row["score_id"]
        task_id = row["task_id"]
        model_name = row["model_name"]
        benchmark = row["benchmark"]
        metrics = row["metrics"]
        completions = row["completions"]
        evals = row["evals"]
        label = next(label for label, pattern in MODELS.items() if pattern.strip("%") in model_name)
        pair = (label, benchmark)
        completed_pairs.add(pair)
        if completions > 0 and completions == evals:
            validated_pairs.add(pair)
        latest.append(
            {
                "score_id": score_id,
                "task_id": task_id,
                "model": label,
                "model_name": model_name,
                "benchmark": benchmark,
                "score": metric_score(metrics),
                "completions": completions,
                "evals": evals,
                "fully_validated": completions > 0 and completions == evals,
                "context_audit": context_audit(metrics),
            }
        )

    missing = [
        {"model": model, "benchmark": benchmark}
        for model in MODELS
        for benchmark in datasets
        if (model, benchmark) not in completed_pairs
    ]
    result = {
        "matrix_total": len(MODELS) * len(datasets),
        "official_score_pairs": len(completed_pairs),
        "fully_validated_pairs": len(validated_pairs),
        "remaining_pairs": len(missing),
        "latest_scores": latest,
        "missing": missing,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
