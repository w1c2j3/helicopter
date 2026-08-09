#!/usr/bin/env python3
"""Thirty-minute health and score audit for the formal EvalScope Agent matrix."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path("/home/rwkv/chase/helicopter-e0eeddc")
PYTHON = ROOT / ".venv/bin/python"
STATUS = Path("/home/rwkv/chase/eval-results/g1h_agent_status.py")
WORKER = Path("/home/rwkv/chase/eval-results/g1h_evalscope_autorun.py")
RUNTIME = Path("/home/rwkv/chase/eval-results/g1h-agent-autorun")
CATALOG = ROOT / "benchmarks/evalscope_agent_datasets.json"
LOCK = RUNTIME / "monitor.lock"
STATE = RUNTIME / "monitor.last_score_id"
LOG = RUNTIME / "monitor.log"

MODEL_LABELS = {
    "1.5B": "rwkv7-g1h-1.5b-20260710-ctx10240",
    "2.9B": "rwkv7-g1h-2.9b-20260710-ctx10240",
    "7.2B": "rwkv7-g1h-7.2b-20260710-ctx10240",
    "13.3B": "rwkv7-g1h-13.3b-20260710-ctx10240",
}
PORTS = {"1.5B": 19415, "2.9B": 19429, "7.2B": 29572, "13.3B": 29533}
LANES = 4
AGGREGATE_ONLY = {"k2_verifier": 2000, "minimax_verifier": 102}
DEFERRED = {"browsecomp"}

# These are the official sample counts observed from complete EvalScope runs.
# The database benchmark.num_samples column is deliberately not used here:
# an interrupted import can overwrite it with a partial count.
EXPECTED_SAMPLES = {
    "bfcl_v3": 4441,
    "browsecomp": 1266,
    "gaia": 165,
    "general_fc": 2000,
    "k2_verifier": 2000,
    "kimi_verifier": 55,
    "minimax_verifier": 102,
    "officeqa": 133,
    "wide_search": 200,
}
def load_remote_env() -> None:
    completed = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "g1h-monitor", "/home/rwkv/chase/helicopter/.env.remote"],
        check=True,
        capture_output=True,
    )
    for item in completed.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            os.environ[key.decode()] = value.decode()


def psql(query: str) -> str:
    environment = os.environ.copy()
    environment["PGPASSWORD"] = environment["SCOREBOARD_DB_PASSWORD"]
    completed = subprocess.run(
        [
            "/usr/bin/psql", "-XAt", "-v", "ON_ERROR_STOP=1",
            "-h", environment.get("SCOREBOARD_DB_HOST", "127.0.0.1"),
            "-p", environment.get("SCOREBOARD_DB_PORT", "55433"),
            "-U", environment["SCOREBOARD_DB_USER"],
            "-d", environment["SCOREBOARD_DB_NAME"], "-c", query,
        ],
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def catalog_names() -> list[str]:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    return [str(row["name"]) for row in payload["datasets"]]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def latest_scores() -> list[dict[str, Any]]:
    datasets = ",".join(sql_literal(item) for item in catalog_names())
    models = ",".join(sql_literal(item) for item in MODEL_LABELS.values())
    query = f"""
      WITH ranked AS (
        SELECT s.score_id, s.task_id, s.metrics, t.status AS task_status,
               m.model_name, b.benchmark_name,
               row_number() OVER (
                 PARTITION BY m.model_name, b.benchmark_name ORDER BY s.score_id DESC
               ) AS rank
        FROM scores s
        JOIN task t ON t.task_id=s.task_id
        JOIN model m ON m.model_id=t.model_id
        JOIN benchmark b ON b.benchmark_id=t.benchmark_id
        WHERE b.benchmark_name IN ({datasets})
          AND m.model_name IN ({models})
      )
      SELECT row_to_json(row) FROM (
        SELECT r.score_id, r.task_id, r.metrics, r.task_status,
               r.model_name, r.benchmark_name,
               count(c.completions_id) AS completions,
               count(DISTINCT c.sample_index) AS distinct_samples,
               min(c.sample_index) AS min_sample_index,
               max(c.sample_index) AS max_sample_index,
               count(c.completions_id) FILTER (WHERE c.status='Completed') AS completed_completions,
               count(c.completions_id) FILTER (WHERE c.status<>'Completed') AS noncompleted_completions,
               count(c.completions_id) FILTER (
                 WHERE c.context IS NULL OR jsonb_typeof(c.context::jsonb)<>'object'
               ) AS bad_context,
               count(c.completions_id) FILTER (
                 WHERE lower(c.context::text) LIKE '%failed to create sandbox%'
                    OR lower(c.context::text) LIKE '%failed to pull image%'
                    OR lower(c.context::text) LIKE '%proxyconnect%'
                    OR lower(c.context::text) LIKE '%connection refused%'
               ) AS suspicious_contexts,
               count(c.completions_id) FILTER (
                 WHERE coalesce(c.context::jsonb #>> '{{agent_result,prediction,model_output,error}}','') <> ''
               ) AS provider_errors,
               count(e.eval_id) AS evals,
               count(e.eval_id) FILTER (WHERE coalesce(btrim(e.answer),'')='') AS empty_answers,
               count(e.eval_id) FILTER (WHERE coalesce(btrim(e.ref_answer),'')='') AS empty_refs,
               count(e.eval_id) FILTER (WHERE e.is_passed) AS passed_evals,
               count(e.eval_id) FILTER (
                 WHERE lower(coalesce(e.fail_reason,'')) LIKE '%proxyconnect%'
                    OR lower(coalesce(e.fail_reason,'')) LIKE '%connection refused%'
                    OR lower(coalesce(e.fail_reason,'')) LIKE '%failed to create sandbox%'
                    OR lower(coalesce(e.fail_reason,'')) LIKE '%failed to pull image%'
               ) AS suspicious_eval_reasons
        FROM ranked r
        LEFT JOIN completions c ON c.task_id=r.task_id
        LEFT JOIN eval e ON e.completions_id=c.completions_id
        WHERE r.rank=1
        GROUP BY r.score_id,r.task_id,r.metrics,r.task_status,r.model_name,r.benchmark_name
        ORDER BY r.score_id
      ) row
    """
    return [json.loads(line) for line in psql(query).splitlines() if line.strip()]


def audit_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    benchmark = str(row["benchmark_name"])
    completions = int(row["completions"] or 0)
    evals = int(row["evals"] or 0)
    expected = AGGREGATE_ONLY.get(benchmark) or EXPECTED_SAMPLES.get(benchmark)
    metrics = row.get("metrics") or {}
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            metrics = {}
    audit = metrics.get("context_audit") if isinstance(metrics, dict) else {}
    if not isinstance(audit, dict):
        audit = {}
    flags: list[str] = []
    if expected is None:
        flags.append("sample_count_unmapped")
    elif completions != expected:
        flags.append(f"sample_count={completions}/{expected}")
    if int(row["distinct_samples"] or 0) != completions:
        flags.append("duplicate_sample_index")
    if completions and (int(row["min_sample_index"] or 0) != 0 or int(row["max_sample_index"] or -1) != completions - 1):
        flags.append("sample_index_range")
    if int(row["bad_context"] or 0):
        flags.append(f"bad_context={row['bad_context']}")
    if int(row["noncompleted_completions"] or 0):
        flags.append(f"noncompleted={row['noncompleted_completions']}")
    if int(row["suspicious_contexts"] or 0):
        flags.append(f"infra_context={row['suspicious_contexts']}")
    if int(row["provider_errors"] or 0):
        flags.append(f"provider_errors={row['provider_errors']}")
    if int(row["suspicious_eval_reasons"] or 0):
        flags.append(f"infra_eval={row['suspicious_eval_reasons']}")
    if row["task_status"] != "Completed":
        flags.append(f"task_status={row['task_status']}")
    if benchmark in AGGREGATE_ONLY:
        if evals:
            flags.append(f"aggregate_has_evals={evals}")
        for key in ("missing_reviews", "invalid_reviews", "context_error_samples", "inference_error_samples"):
            if int(audit.get(key) or 0):
                flags.append(f"{key}={audit[key]}")
        if int(audit.get("samples") or 0) != completions:
            flags.append("audit_sample_mismatch")
    else:
        if evals != completions:
            flags.append(f"eval_mismatch={evals}/{completions}")
        if int(audit.get("samples") or 0) != completions:
            flags.append("audit_sample_mismatch")
        for key in ("missing_reviews", "invalid_reviews", "context_error_samples", "inference_error_samples"):
            if int(audit.get(key) or 0):
                flags.append(f"{key}={audit[key]}")
    score = metrics.get("score") if isinstance(metrics, dict) else None
    try:
        numeric_score = float(score)
        if not 0.0 <= numeric_score <= 1.0:
            flags.append(f"score_range={numeric_score}")
    except (TypeError, ValueError):
        flags.append("score_missing_or_non_numeric")
    return not flags, flags


def endpoint_ok(port: int) -> bool:
    completed = subprocess.run(
        ["curl", "-fsS", "--max-time", "10", "-H", "Authorization: Bearer rwkv-skills", f"http://127.0.0.1:{port}/v1/models"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def worker_name(model: str, lane: int) -> str:
    return f"g1h-agent-{model.replace('.', '').replace('B', 'b').lower()}-lane-{lane}"


def worker_present(model: str, lane: int) -> bool:
    pattern = f"g1h_evalscope_autorun.py worker {model} --lane {lane} --lanes {LANES}"
    return subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False).stdout.strip() != ""


def start_worker(model: str, lane: int) -> None:
    name = worker_name(model, lane)
    logfile = RUNTIME / "logs" / f"worker-{model}-{lane}.screen.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "screen", "-dmS", name, "-L", "-Logfile", str(logfile),
        "bash", "-lc", f"exec {PYTHON} {WORKER} worker {model} --lane {lane} --lanes {LANES}",
    ]
    subprocess.run(command, check=True)


def ensure_workers(endpoints: dict[str, bool], has_pending: bool) -> list[str]:
    if not has_pending:
        return []
    started: list[str] = []
    for model, healthy in endpoints.items():
        if not healthy:
            continue
        for lane in range(LANES):
            if not worker_present(model, lane):
                start_worker(model, lane)
                started.append(f"{model}/lane-{lane}")
    return started


def proxy_ok(port: int) -> bool:
    completed = subprocess.run(
        ["curl", "-fsS", "--max-time", "10", "-o", "/dev/null", "-x", f"http://127.0.0.1:{port}", "https://docker.m.daocloud.io/v2/"],
        check=False,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0 or completed.returncode == 22


def write_log(lines: list[str]) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def monitor_once() -> int:
    load_remote_env()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        rows = latest_scores()
        previous = int(STATE.read_text(encoding="utf-8").strip()) if STATE.is_file() else 0
        max_score_id = max((int(row["score_id"]) for row in rows), default=previous)
        new_score_ids = sorted(int(row["score_id"]) for row in rows if int(row["score_id"]) > previous)
        audited: list[tuple[dict[str, Any], bool, list[str]]] = []
        for row in rows:
            if row["benchmark_name"] in DEFERRED:
                continue
            trusted, flags = audit_row(row)
            audited.append((row, trusted, flags))
        endpoints = {model: endpoint_ok(port) for model, port in PORTS.items()}
        workers = {
            f"{model}/lane-{lane}": worker_present(model, lane)
            for model in MODEL_LABELS
            for lane in range(LANES)
        }
        active_benchmarks = set(catalog_names()) - DEFERRED
        scored_active_pairs = {
            (row["model_name"], row["benchmark_name"])
            for row in rows
            if row["benchmark_name"] in active_benchmarks
        }
        pending = (
            any(not trusted for _, trusted, _ in audited)
            or len(scored_active_pairs) < len(MODEL_LABELS) * len(active_benchmarks)
        )
        started = ensure_workers(endpoints, pending)
        if started:
            time.sleep(1)
            workers = {
                f"{model}/lane-{lane}": worker_present(model, lane)
                for model in MODEL_LABELS
                for lane in range(LANES)
            }
        usage = shutil.disk_usage("/")
        disk_pct = usage.used / usage.total * 100.0
        proxy_state = {str(port): proxy_ok(port) for port in (31080, 31081)}
        trusted_count = sum(trusted for _, trusted, _ in audited)
        issues: list[str] = []
        issues.extend(f"endpoint_down={model}" for model, healthy in endpoints.items() if not healthy)
        issues.extend(f"worker_missing={name}" for name, present in workers.items() if not present)
        issues.extend(f"proxy_unhealthy={port}" for port, healthy in proxy_state.items() if not healthy)
        if disk_pct >= 90.0:
            issues.append(f"disk={disk_pct:.1f}%")
        for row, trusted, flags in audited:
            if not trusted:
                issues.append(f"untrusted={row['model_name']}:{row['benchmark_name']}:{','.join(flags)}")
        lines = [
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} scores={len(rows)} trusted={trusted_count} new_score_ids={','.join(map(str,new_score_ids)) or '-'} pending={pending} deferred={','.join(sorted(DEFERRED)) or '-'}",
            f"endpoints={endpoints} proxies={proxy_state} workers_present={sum(workers.values())}/{len(workers)} disk={disk_pct:.1f}% started={','.join(started) or '-'}",
        ]
        lines.extend(f"score task={row['task_id']} {row['model_name']}:{row['benchmark_name']} score={((row.get('metrics') or {}).get('score') if isinstance(row.get('metrics'), dict) else None)} trusted={trusted} flags={','.join(flags) or '-'}" for row, trusted, flags in audited)
        lines.extend(f"ISSUE {issue}" for issue in issues)
        write_log(lines)
        STATE.write_text(str(max_score_id) + "\n", encoding="utf-8")
        return 1 if issues else 0


def main() -> int:
    return monitor_once()


if __name__ == "__main__":
    raise SystemExit(main())
