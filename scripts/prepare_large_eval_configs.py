"""Apply the common rollout policy to the curated large-eval suite.

This edits the [evaluation] and [sampling] policy keys in the 60 benchmark
TOMLs.  The metric and avg_k declared by each benchmark are preserved: avg
tasks use the TOML-selected avg@k and native tasks keep their native scorer
and receive four rollouts for its own aggregation.

Sampling is deliberately field-specific.  Math follows the local RWKV math
evaluation family (temperature 0.8 with the math top-p/presence settings),
while knowledge, coding, and instruction-following use the repository's
general official-style sampling family.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/large_eval_60.toml"

SAMPLING_BY_FIELD: dict[str, dict[str, str]] = {
    "math": {
        "temperature": "0.8",
        "top_p": "0.28",
        "top_k": "32",
        "presence_penalty": "0.0",
        "frequency_penalty": "0.0",
        "penalty_decay": "1.0",
    },
    "knowledge": {
        "temperature": "0.96",
        "top_p": "0.76",
        "top_k": "32",
        "presence_penalty": "1.0",
        "frequency_penalty": "0.1",
        "penalty_decay": "0.988",
    },
    "coding": {
        "temperature": "0.96",
        "top_p": "0.76",
        "top_k": "32",
        "presence_penalty": "1.0",
        "frequency_penalty": "0.1",
        "penalty_decay": "0.988",
    },
    "instruction_following": {
        "temperature": "0.96",
        "top_p": "0.76",
        "top_k": "32",
        "presence_penalty": "1.0",
        "frequency_penalty": "0.1",
        "penalty_decay": "0.988",
    },
}


def _section(text: str, name: str) -> tuple[int, int, str]:
    match = re.search(
        rf"(?ms)^\[{re.escape(name)}\]\s*\n(.*?)(?=^\[|\Z)",
        text,
    )
    if not match:
        raise ValueError(f"missing [{name}] section")
    return match.start(1), match.end(1), match.group(1)


def _set_key(section: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*.*$")
    replacement = f"{key} = {value}"
    if pattern.search(section):
        return pattern.sub(replacement, section, count=1)
    return section.rstrip() + f"\n{replacement}\n"


def main() -> int:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    benchmarks = manifest.get("benchmarks", [])
    if len(benchmarks) != 60:
        raise SystemExit(f"expected exactly 60 benchmarks, found {len(benchmarks)}")

    changed: list[str] = []
    for entry in benchmarks:
        relative = Path(str(entry["config"]))
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"benchmark config not found: {relative}")
        text = path.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        benchmark = parsed.get("benchmark", {})
        if str(benchmark.get("task")) != str(entry["task"]):
            raise SystemExit(
                f"task mismatch in {relative}: {benchmark.get('task')!r} != {entry['task']!r}"
            )
        start, end, evaluation = _section(text, "evaluation")
        sampling_start, sampling_end, sampling = _section(text, "sampling")
        field = str(entry.get("field", "")).strip()
        if field not in SAMPLING_BY_FIELD:
            raise SystemExit(f"unsupported benchmark field {field!r} in {relative}")
        metric = str(parsed.get("evaluation", {}).get("metric", "avg")).strip().lower()
        if metric not in {"avg", "native"}:
            raise SystemExit(f"unsupported metric {metric!r} in {relative}")
        configured_avg_k = int(parsed.get("evaluation", {}).get("avg_k", 4))
        if configured_avg_k <= 0:
            raise SystemExit(f"invalid avg_k {configured_avg_k} in {relative}")
        updates = {
            "metric": f'"{metric}"',
            "avg_k": str(configured_avg_k if metric == "avg" else 1),
            "rollout_n": str(configured_avg_k if metric == "avg" else 4),
            "pass_k": "1",
            "pass_n": "4",
        }
        updated = evaluation
        for key, value in updates.items():
            updated = _set_key(updated, key, value)
        updated_sampling = sampling
        for key, value in SAMPLING_BY_FIELD[field].items():
            updated_sampling = _set_key(updated_sampling, key, value)
        if updated != evaluation or updated_sampling != sampling:
            rewritten = (
                text[:start]
                + updated
                + text[end:sampling_start]
                + updated_sampling
                + text[sampling_end:]
            )
            path.write_text(rewritten, encoding="utf-8")
            changed.append(str(relative))

    print(f"updated {len(changed)} benchmark configs")
    for item in changed:
        print(item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
