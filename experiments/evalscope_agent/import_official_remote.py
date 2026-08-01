from __future__ import annotations

import argparse
from pathlib import Path

from helicopter_cli.evalscope_scoreboard import (
    build_import_plan,
    cleanup_json_artifacts,
    persist_import_plan_sync,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model")
    parser.add_argument("benchmark")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    root = Path("/home/rwkv/chase/EvalScope")
    plan = build_import_plan(args.work_dir, model_name=args.model, benchmark=args.benchmark)
    result = persist_import_plan_sync(plan, root=root)
    print(result)
    print(plan.context_audit)
    if args.cleanup:
        print(f"removed_json={cleanup_json_artifacts(args.work_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
