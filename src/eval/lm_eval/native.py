from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import yaml


RWKV_ONLY_FIELDS = frozenset(
    {
        "backend",
        "benchmark_configs",
        "eot_token_id",
        "generation_kwargs",
        "max_gen_toks",
        "output_dir",
        "pool_manifests",
        "prompt",
        "publish",
        "schema_version",
        "task_include_paths",
        "weights",
        "wkv_modes",
    }
)


class NativeConfigError(ValueError):
    pass


def run(*, config: Mapping[str, object], dry_run: bool) -> int:
    rwkv_fields = sorted(RWKV_ONLY_FIELDS.intersection(config))
    if rwkv_fields:
        raise NativeConfigError(
            "native lm-eval config must not use RWKV-only fields: "
            + ", ".join(rwkv_fields)
        )

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml"
    ) as stream:
        yaml.safe_dump(dict(config), stream, allow_unicode=True, sort_keys=False)
        stream.flush()
        native_config = Path(stream.name)
        if dry_run:
            return _dry_run(native_config)
        completed = subprocess.run(
            [sys.executable, "-m", "lm_eval", "run", "--config", str(native_config)],
            check=False,
        )
        return completed.returncode


def _dry_run(config_path: Path) -> int:
    from lm_eval.config.evaluate_config import EvaluatorConfig

    try:
        config = EvaluatorConfig.from_config(config_path)
        task_manager = config.process_tasks(config.metadata)
    except (TypeError, ValueError) as error:
        raise NativeConfigError(str(error)) from error

    resolved_tasks = [
        task if isinstance(task, str) else task.get("task", "<custom task>")
        for task in config.tasks
    ]
    print(
        json.dumps(
            {
                "status": "ready",
                "route": "native",
                "evaluator": {
                    "name": "lm-eval",
                    "version": importlib.metadata.version("lm-eval"),
                },
                "model": config.model,
                "output_path": config.output_path,
                "resolved_tasks": resolved_tasks,
                "task_count": len(resolved_tasks),
                "task_manager": type(task_manager).__name__,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0
