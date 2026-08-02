from __future__ import annotations

import importlib.metadata
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Mapping

from helicopter_lighteval.http_pool import PoolError, VLLMHttpPool

from .config import ConfigError, LMEvalConfig
from .model import RWKVVLLMHttpLM


def run(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    pool: VLLMHttpPool | None = None
    try:
        config = LMEvalConfig.read(config_path, env)
        task_manager, resolved_tasks = _resolve_tasks(config.tasks)
        pool = VLLMHttpPool(config.manifest)
        model_id = pool.preflight()
        version = importlib.metadata.version("lm-eval")
        if version != "0.4.12":
            raise ConfigError(f"lm-eval version must be 0.4.12, found {version}")
        if dry_run:
            print(
                json.dumps(
                    {
                        "status": "ready",
                        "evaluator": {"name": "lm-eval", "version": version},
                        "config": config.public(),
                        "model_id": model_id,
                        "global_step": config.manifest.global_step,
                        "wkv_mode": config.manifest.wkv_mode,
                        "max_model_len": config.manifest.max_model_len,
                        "effective_max_length": config.manifest.max_model_len - 2,
                        "resolved_tasks": list(resolved_tasks),
                        "task_selection": {
                            "selectors": list(config.tasks),
                            "resolved": _describe_tasks(
                                task_manager, resolved_tasks
                            ),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        model = RWKVVLLMHttpLM(
            pool=pool,
            eot_token_id=config.eot_token_id,
            batch_size=config.batch_size,
            max_gen_toks=config.max_gen_toks,
        )
        import lm_eval

        results = lm_eval.simple_evaluate(
            model=model,
            tasks=list(resolved_tasks),
            batch_size=config.batch_size,
            limit=config.limit,
            log_samples=config.log_samples,
            task_manager=task_manager,
        )
        if results is None:
            raise RuntimeError("lm-eval returned no results")
        _write_results(
            config=config,
            model_id=model_id,
            version=version,
            results=results,
        )
        print(f"lm-eval results written to {config.output_dir}")
        return 0
    except (ConfigError, PoolError) as error:
        raise SystemExit(str(error)) from error
    except Exception as error:
        name = f"{type(error).__module__}.{type(error).__qualname__}"
        detail = str(error).strip()
        suffix = f": {detail}" if detail else ""
        raise SystemExit(f"lm-eval evaluation failed: {name}{suffix}") from error
    finally:
        if pool is not None:
            pool.close()


def _resolve_tasks(tasks: tuple[str, ...]):
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    resolved: list[str] = []
    for selector in tasks:
        matches = manager.match_tasks([selector])
        if not matches:
            raise ConfigError(f"unknown lm-eval task, group, tag, or pattern: {selector}")
        for task_name in matches:
            if task_name not in resolved:
                resolved.append(task_name)
    return manager, tuple(resolved)


def _describe_tasks(manager, tasks: tuple[str, ...]) -> list[dict[str, object]]:
    descriptions: list[dict[str, object]] = []
    for task_name in tasks:
        entry = manager.task_index[task_name]
        config = entry.cfg or {}
        descriptions.append(
            {
                "name": task_name,
                "kind": entry.kind.name.lower(),
                "output_type": config.get("output_type"),
            }
        )
    return descriptions


def _write_results(
    *,
    config: LMEvalConfig,
    model_id: str,
    version: str,
    results: Mapping[str, object],
) -> None:
    from lm_eval.utils import handle_non_serializable

    _prepare_output_dir(config.output_dir)
    raw = json.loads(
        json.dumps(results, default=handle_non_serializable, ensure_ascii=False)
    )
    summary = {
        "schema_version": 1,
        "evaluator": {"name": "lm-eval", "version": version},
        "backend": "vllm_http",
        "model_id": model_id,
        "global_step": config.manifest.global_step,
        "wkv_mode": config.manifest.wkv_mode,
        "max_model_len": config.manifest.max_model_len,
        "effective_max_length": config.manifest.max_model_len - 2,
        "tasks": list(config.tasks),
        "metrics": raw.get("results", {}),
        "versions": raw.get("versions", {}),
    }
    _atomic_json(config.output_dir / "results.json", raw)
    _atomic_json(config.output_dir / "summary.json", summary)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.geteuid()
    ):
        raise ConfigError("output_dir must be an owned regular directory")
    path.chmod(0o700)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)
