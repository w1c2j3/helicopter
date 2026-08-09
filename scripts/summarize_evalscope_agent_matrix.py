#!/usr/bin/env python3
"""Summarize complete EvalScope Agent acceptance reports.

A row is complete only when the CLI exited successfully, an official report
exists, and its sample count equals the local acceptance sample count.
Incomplete runs remain visible but never contribute a score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _official_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.get("official_reports", []):
        if not isinstance(item, dict) or not isinstance(item.get("report"), dict):
            continue
        official = item["report"]
        metrics = {
            str(metric["name"]): metric.get("score")
            for metric in official.get("metrics", [])
            if isinstance(metric, dict) and metric.get("name")
        }
        rows.append(
            {
                "path": item.get("path"),
                "dataset": official.get("dataset_name"),
                "model": official.get("model_name"),
                "score": official.get("score"),
                "num": official.get("num"),
                "metrics": metrics,
            }
        )
    return rows


def summarize_matrix(roots: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("acceptance_report.json")):
            report = _read(path)
            if report is None:
                continue
            official = _official_rows(report)
            samples = report.get("samples")
            sample_count = len(samples) if isinstance(samples, list) else 0
            counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
            nums = [item["num"] for item in official if isinstance(item.get("num"), int)]
            official_count = sum(nums) if nums else None
            complete = (
                report.get("exit_code") == 0
                and bool(official)
                and sample_count > 0
                and official_count == sample_count
            )
            rows.append(
                {
                    "status": "complete" if complete else "incomplete",
                    "report": str(path),
                    "exit_code": report.get("exit_code"),
                    "model": official[0].get("model") if official else None,
                    "dataset": official[0].get("dataset") if official else None,
                    "score": official[0].get("score") if len(official) == 1 else None,
                    "sample_count": sample_count,
                    "official_sample_count": official_count,
                    "official_reports": official,
                    "local_counts": counts,
                }
            )
    return {
        "roots": [str(root) for root in roots],
        "complete": [row for row in rows if row["status"] == "complete"],
        "incomplete": [row for row in rows if row["status"] != "complete"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="+", type=Path, help="matrix result roots to scan")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    summary = summarize_matrix(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    for row in summary["complete"]:
        print(
            f"complete\t{row['model']}\t{row['dataset']}\t{row['score']}\t"
            f"{row['sample_count']}\t{row['report']}"
        )
    for row in summary["incomplete"]:
        print(
            f"incomplete\t{row['model']}\t{row['dataset']}\t"
            f"{row['exit_code']}\t{row['report']}"
        )
    print(
        f"complete={len(summary['complete'])} "
        f"incomplete={len(summary['incomplete'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
