from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .commands import (
    CommandPlan,
    build_infer_plan,
    is_local_base_url,
    local_openai_base_url,
)
from .config import resolve_model_entry, table
from .env import env_value, pick
from .eval_run import (
    DEFAULT_SERVER_TIMEOUT_S,
    format_plan_for_display,
    infer_args_namespace,
    port_from_base_url,
    server_is_healthy,
    stop_server,
    wait_for_server,
)
from .naive_chat_proxy import NaiveChatProxy
from .evalscope_agent_results import write_trace_report
from .paths import resolve_path
from .runner import run_command


DEFAULT_CATALOG = "benchmarks/evalscope_agent_datasets.json"
DEFAULT_OUTPUT_DIR = "results/evalscope"
DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "strategy": "function_calling",
    "tools": ["bash"],
    "environment": "docker",
    "max_steps": 10,
}


def _catalog_path(root: Path, value: str | None) -> Path:
    path = Path(value or DEFAULT_CATALOG)
    return path if path.is_absolute() else root / path


def load_agent_catalog(root: Path, value: str | None = None) -> list[dict[str, Any]]:
    path = _catalog_path(root, value)
    if not path.is_file():
        raise SystemExit(f"EvalScope agent dataset catalog not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid EvalScope agent dataset catalog: {path}: {error}") from error
    rows = raw.get("datasets") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: datasets must be a JSON array")
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("name")]


def format_agent_catalog(rows: list[dict[str, Any]], output_format: str = "text") -> str:
    if output_format == "json":
        return json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if output_format == "summary":
        return f"total\t{len(rows)}\n" + "".join(
            f"{row['name']}\t{row.get('display_name', row['name'])}\n" for row in rows
        )
    lines = ["dataset\tdisplay_name\tcategories"]
    lines.extend(
        "\t".join(
            (
                str(row["name"]),
                str(row.get("display_name") or row["name"]),
                ",".join(str(item) for item in row.get("categories", []) or []),
            )
        )
        for row in rows
    )
    return "\n".join(lines) + "\n"


def _output_dir(config: dict[str, Any], *, root: Path, env: dict[str, str], args: Any) -> Path:
    settings = table(config, "evalscope")
    value = pick(
        getattr(args, "work_dir", None),
        env_value(env, "HELICOPTER_EVALSCOPE_OUTPUT_DIR"),
        settings.get("output_dir"),
        settings.get("work_dir"),
        DEFAULT_OUTPUT_DIR,
    )
    return resolve_path(str(value), root=root, env=env)


def _json_mapping(value: Any, *, root: Path, name: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise SystemExit(f"{name} must be a JSON object or TOML table")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    try:
        parsed = json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{name} must be valid JSON or a readable JSON file: {value}") from error
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return dict(parsed)


def _datasets(args: Any, config: dict[str, Any]) -> list[str]:
    settings = table(config, "evalscope")
    raw = getattr(args, "datasets", None)
    if not raw:
        raw = settings.get("datasets")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise SystemExit("no EvalScope datasets given: pass dataset names or set [evalscope].datasets")
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not values:
        raise SystemExit("no EvalScope datasets given: pass dataset names or set [evalscope].datasets")
    return values


def _tools(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, list):
        values = value
    else:
        raise SystemExit("EvalScope agent tools must be a list or comma-separated string")
    result: list[str] = []
    for item in values:
        result.extend(part.strip() for part in str(item).split(",") if part.strip())
    return result


def _agent_config(args: Any, *, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    settings = table(config, "evalscope")
    mode = str(pick(getattr(args, "mode", None), settings.get("mode"), "native")).strip().lower()
    if mode not in {"native", "bridge"}:
        raise SystemExit("EvalScope agent mode must be 'native' or 'bridge'")

    configured = _json_mapping(settings.get("agent_config"), root=root, name="[evalscope].agent_config")
    override = _json_mapping(getattr(args, "agent_config", None), root=root, name="--agent-config")
    if mode == "native":
        result = {**DEFAULT_AGENT_CONFIG, **configured, **override}
        if getattr(args, "strategy", None):
            result["strategy"] = args.strategy
        tools = _tools(getattr(args, "tools", None))
        if tools is not None:
            result["tools"] = tools
        if getattr(args, "agent_environment", None):
            result["environment"] = args.agent_environment
        if getattr(args, "max_steps", None) is not None:
            result["max_steps"] = args.max_steps
        return result

    result = {
        "framework": str(
            pick(
                getattr(args, "framework", None),
                override.get("framework"),
                configured.get("framework"),
                settings.get("framework"),
                "mock",
            )
        ),
        "environment": str(
            pick(
                getattr(args, "agent_environment", None),
                override.get("environment"),
                configured.get("environment"),
                "local",
            )
        ),
        **configured,
        **override,
    }
    if getattr(args, "agent_timeout", None) is not None:
        result["timeout"] = args.agent_timeout
    return result


def _config_json(
    args: Any,
    *,
    root: Path,
    settings: dict[str, Any],
    arg_name: str,
    config_name: str,
) -> dict[str, Any]:
    value = getattr(args, arg_name, None)
    if value is None:
        value = settings.get(config_name)
    return _json_mapping(value, root=root, name=f"--{arg_name.replace('_', '-')} / [evalscope].{config_name}")


def _append_json(command: list[str], flag: str, value: dict[str, Any]) -> None:
    if value:
        command.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])


def build_evalscope_plan(
    args: Any,
    *,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
    api_url: str | None = None,
) -> CommandPlan:
    if not getattr(args, "model", None):
        raise SystemExit("eval evalscope requires a model alias")
    settings = table(config, "evalscope")
    datasets = _datasets(args, config)
    model = resolve_model_entry(config, args.model)
    served_model_name = str(
        pick(
            getattr(args, "served_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )
    base_url = api_url or local_openai_base_url(config, env, args)
    api_key = pick(
        getattr(args, "api_key", None),
        model.get("api_key"),
        env_value(env, "HELICOPTER_EVAL_API_KEY", "OPENAI_API_KEY"),
        settings.get("api_key"),
        "EMPTY" if is_local_base_url(base_url) else None,
    )
    output_dir = _output_dir(config, root=root, env=env, args=args)
    binary = str(pick(getattr(args, "binary", None), settings.get("binary"), "evalscope"))
    eval_type = str(pick(getattr(args, "eval_type", None), settings.get("eval_type"), "openai_api"))
    command = [binary, "eval", "--model", served_model_name, "--api-url", base_url, "--eval-type", eval_type]
    if api_key:
        command.extend(["--api-key", str(api_key)])
    command.extend(["--datasets", *datasets, "--work-dir", str(output_dir)])

    limit = pick(getattr(args, "limit", None), settings.get("limit"))
    if limit is not None:
        command.extend(["--limit", str(limit)])
    eval_batch_size = pick(getattr(args, "eval_batch_size", None), settings.get("eval_batch_size"))
    if eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(eval_batch_size)])
    dataset_hub = pick(getattr(args, "dataset_hub", None), settings.get("dataset_hub"))
    if dataset_hub:
        command.extend(["--dataset-hub", str(dataset_hub)])
    generation_config = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="generation_config",
        config_name="generation_config",
    )
    _append_json(command, "--generation-config", generation_config)
    dataset_args = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="dataset_args",
        config_name="dataset_args",
    )
    _append_json(command, "--dataset-args", dataset_args)
    sandbox = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="sandbox",
        config_name="sandbox",
    )
    _append_json(command, "--sandbox", sandbox)
    command.extend(["--agent-config", json.dumps(_agent_config(args, root=root, config=config), ensure_ascii=False, separators=(",", ":"))])
    if pick(getattr(args, "no_timestamp", None), settings.get("no_timestamp"), False):
        command.append("--no-timestamp")
    use_cache = pick(getattr(args, "use_cache", None), settings.get("use_cache"))
    if use_cache:
        command.extend(["--use-cache", str(use_cache)])
    if pick(getattr(args, "rerun_review", None), settings.get("rerun_review"), False):
        command.append("--rerun-review")
    if pick(getattr(args, "enable_progress_tracker", None), settings.get("enable_progress_tracker"), False):
        command.append("--enable-progress-tracker")
    collect_perf = pick(getattr(args, "collect_perf", None), settings.get("collect_perf"))
    if collect_perf is True:
        command.append("--collect-perf")
    elif collect_perf is False:
        command.append("--no-collect-perf")
    if pick(getattr(args, "debug", None), settings.get("debug"), False):
        command.append("--debug")
    if pick(getattr(args, "ignore_errors", None), settings.get("ignore_errors"), False):
        command.append("--ignore-errors")
    for extra in settings.get("extra_args", []) or []:
        if not isinstance(extra, str):
            raise SystemExit("[evalscope].extra_args must be a TOML string array")
        command.append(extra)
    shown_env = {"EVALSCOPE_BINARY": binary}
    return CommandPlan(command=command, cwd=root, shown_env=shown_env, env=dict(env))


def _infer_plan(args: Any, *, root: Path, env: dict[str, str], config: dict[str, Any], base_url: str) -> CommandPlan | None:
    if getattr(args, "no_server", False) or not is_local_base_url(base_url):
        return None
    return build_infer_plan(
        infer_args_namespace(args, port=port_from_base_url(base_url)),
        root=root,
        env=env,
        config=config,
    )


def run_evalscope(args: Any, *, root: Path, env: dict[str, str], config: dict[str, Any]) -> int:
    if getattr(args, "list_datasets", False):
        rows = load_agent_catalog(root, getattr(args, "dataset_catalog", None))
        print(format_agent_catalog(rows, getattr(args, "format", "text")), end="")
        return 0
    base_url = local_openai_base_url(config, env, args)
    settings = table(config, "evalscope")
    output_dir = _output_dir(config, root=root, env=env, args=args)
    use_naive_proxy = bool(
        pick(
            getattr(args, "naive_chat_proxy", None),
            settings.get("naive_chat_proxy"),
            True,
        )
    )
    plan = build_evalscope_plan(args, root=root, env=env, config=config)
    infer_plan = _infer_plan(args, root=root, env=env, config=config, base_url=base_url)
    if getattr(args, "dry_run", False):
        if infer_plan is not None:
            print(format_plan_for_display(infer_plan))
        print(format_plan_for_display(plan))
        return 0

    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    proxy: NaiveChatProxy | None = None
    run_exit_code: int | None = None
    if infer_plan is not None:
        if server_is_healthy(base_url):
            print(f"evalscope: reusing healthy server at {base_url}")
        else:
            server_log = output_dir / "server_logs" / "vllm.log"
            server_log.parent.mkdir(parents=True, exist_ok=True)
            with server_log.open("wb") as log_file:
                server_process = subprocess.Popen(
                    infer_plan.command,
                    cwd=str(infer_plan.cwd),
                    env=infer_plan.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            wait_for_server(
                base_url,
                process=server_process,
                log_path=server_log,
                timeout_s=float(getattr(args, "server_timeout", None) or DEFAULT_SERVER_TIMEOUT_S),
            )
            print(f"evalscope: server healthy at {base_url}")
    try:
        if use_naive_proxy:
            api_key = pick(
                getattr(args, "api_key", None),
                env_value(env, "HELICOPTER_EVAL_API_KEY", "OPENAI_API_KEY"),
                settings.get("api_key"),
                "EMPTY",
            )
            proxy = NaiveChatProxy(
                base_url,
                api_key=str(api_key),
                trace_path=output_dir / "raw" / "naive_chat.jsonl",
            )
            proxy.start()
            plan = build_evalscope_plan(
                args,
                root=root,
                env=env,
                config=config,
                api_url=proxy.base_url,
            )
            print(f"evalscope: naive Chat proxy listening at {proxy.base_url}; upstream {base_url}")
        run_exit_code = run_command(plan.command, cwd=plan.cwd, env=plan.env, shown_env=plan.shown_env, dry_run=False)
        return run_exit_code
    finally:
        if proxy is not None:
            proxy.close()
            trace_report = output_dir / "raw" / "trace_report.json"
            write_trace_report(proxy.trace_path, trace_report, exit_code=run_exit_code)
            print(f"evalscope: raw request/response trace report written to {trace_report}")
        if server_process is not None and not getattr(args, "keep_server", False):
            print("evalscope: stopping vLLM server")
            stop_server(server_process)
        elif server_process is not None:
            print(f"evalscope: leaving vLLM server running (pid {server_process.pid})")
