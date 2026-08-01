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
    SCOREBOARD_LOCK,
    _scoreboard_env,
    format_plan_for_display,
    infer_args_namespace,
    port_from_base_url,
    server_is_healthy,
    stop_server,
    wait_for_server,
)
from .naive_chat_proxy import NaiveChatProxy
from .parallel_candidate_proxy import ParallelCandidateConfig, ParallelCandidateProxy
from .evalscope_agent_results import write_acceptance_report, write_trace_report
from .evalscope_scoreboard import (
    build_import_plan,
    cleanup_json_artifacts,
    persist_import_plan_sync,
)
from .paths import resolve_path
from .runner import run_command


DEFAULT_CATALOG = "benchmarks/evalscope_agent_datasets.json"
DEFAULT_OUTPUT_DIR = "results/evalscope"
# RWKV Agent requests can legitimately take several minutes with a full
# context.  Keep a finite default so one stalled HTTP response cannot hold a
# full benchmark indefinitely; callers can override it or explicitly set
# generation_config.timeout to null when they need the legacy behavior.
DEFAULT_REQUEST_TIMEOUT_S = 600.0
DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "strategy": "function_calling",
    "tools": ["bash"],
    "environment": "docker",
    "max_steps": 10,
}
_LOCAL_EVALSCOPE_PROXY_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _evalscope_child_env(environment: dict[str, str], api_url: str) -> dict[str, str]:
    """Prepare the EvalScope child environment for a local model endpoint.

    httpx initializes a proxy transport from ``all_proxy`` even when the
    request target is covered by ``NO_PROXY``.  A local EvalScope run must not
    fail before the first model request because a desktop SOCKS proxy is
    present in the inherited tmux environment.  Remote endpoints retain the
    caller's environment unchanged.
    """

    child = dict(environment)
    if not is_local_base_url(api_url):
        return child
    for key in _LOCAL_EVALSCOPE_PROXY_ENV:
        child.pop(key, None)
    no_proxy_values = {
        item.strip()
        for item in str(child.get("NO_PROXY") or child.get("no_proxy") or "").split(",")
        if item.strip()
    }
    no_proxy_values.update({"localhost", "127.0.0.1", "::1"})
    no_proxy = ",".join(sorted(no_proxy_values))
    child["NO_PROXY"] = no_proxy
    child["no_proxy"] = no_proxy
    return child


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


def _latest_evalscope_work_dir(output_dir: Path) -> Path:
    """Resolve EvalScope's timestamped child directory when it creates one."""

    if any((output_dir / name).is_dir() for name in ("predictions", "reviews", "reports")):
        return output_dir
    candidates: list[Path] = []
    if output_dir.is_dir():
        for candidate in output_dir.iterdir():
            if candidate.is_dir() and any((candidate / name).is_dir() for name in ("predictions", "reviews", "reports")):
                candidates.append(candidate)
    if not candidates:
        return output_dir
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _saved_trace_exit_code(path: Path) -> int | None:
    """Read a prior run's exit code without treating a missing trace as success."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("exit_code")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _served_model_name(args: Any, *, config: dict[str, Any]) -> str:
    model = resolve_model_entry(config, args.model)
    return str(
        pick(
            getattr(args, "served_model_name", None),
            model.get("served_model_name"),
            model.get("requested_name"),
            args.model,
        )
    )


def _persist_evalscope_scoreboard(
    *,
    work_dir: Path,
    args: Any,
    root: Path,
    env: dict[str, str],
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    """Import official EvalScope artifacts and optionally remove JSON files."""

    try:
        model_name = _served_model_name(args, config=config)
        try:
            benchmarks = _datasets(args, config)
        except SystemExit:
            # ``--report-only`` can be used against the historical BFCL run
            # without repeating its positional dataset argument.
            benchmarks = ["bfcl_v4"]
        results: list[str] = []
        plans = []
        for benchmark in benchmarks:
            plan = build_import_plan(work_dir, model_name=model_name, benchmark=benchmark)
            with SCOREBOARD_LOCK, _scoreboard_env(env):
                result = persist_import_plan_sync(plan, root=root)
            plans.append(plan)
            results.append(result)
            print(
                "evalscope: official results persisted to scoreboard: "
                f"{result}; context_audit={json.dumps(plan.context_audit, ensure_ascii=False, sort_keys=True)}"
            )
        if getattr(args, "scoreboard_db_only", False):
            removed = cleanup_json_artifacts(work_dir)
            print(f"evalscope: removed {removed} JSON/JSONL artifacts after verified DB import")
        return True, "; ".join(results)
    except Exception as error:  # noqa: BLE001 - surface DB failure without hiding the official run result
        print(f"evalscope: scoreboard import failed; JSON artifacts were retained: {error}")
        return False, None


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
    if mode not in {"native", "bridge", "external"}:
        raise SystemExit("EvalScope agent mode must be 'native', 'external', or the legacy alias 'bridge'")

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

    # EvalScope 1.9.x calls this discriminator ``external``.  Keep accepting
    # the project's historical ``bridge`` spelling, but emit only fields from
    # ExternalAgentConfig; forwarding the native strategy/tools/max_steps table
    # makes Pydantic reject the entire task before any request is sent.
    result: dict[str, Any] = {
        "mode": "external",
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
    }
    timeout = pick(
        getattr(args, "agent_timeout", None),
        override.get("timeout"),
        configured.get("timeout"),
    )
    if timeout is not None:
        result["timeout"] = timeout
    for field_name in ("bridge", "environment_extra", "kwargs", "skills_dir", "skill_prompt_nudge"):
        value = pick(override.get(field_name), configured.get(field_name))
        if value is not None:
            result[field_name] = value
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
    request_timeout = pick(
        getattr(args, "request_timeout", None),
        settings.get("request_timeout"),
        DEFAULT_REQUEST_TIMEOUT_S,
    )
    if "timeout" not in generation_config:
        try:
            request_timeout = float(request_timeout)
        except (TypeError, ValueError) as error:
            raise SystemExit("EvalScope request timeout must be a positive number") from error
        if request_timeout <= 0:
            raise SystemExit("EvalScope request timeout must be a positive number")
        generation_config["timeout"] = request_timeout
    _append_json(command, "--generation-config", generation_config)
    model_args = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="model_args",
        config_name="model_args",
    )
    if "timeout" not in model_args and generation_config.get("timeout") is not None:
        model_args["timeout"] = generation_config["timeout"]
    _append_json(command, "--model-args", model_args)
    dataset_args = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="dataset_args",
        config_name="dataset_args",
    )
    _append_json(command, "--dataset-args", dataset_args)
    judge_strategy = pick(getattr(args, "judge_strategy", None), settings.get("judge_strategy"))
    if judge_strategy:
        command.extend(["--judge-strategy", str(judge_strategy)])
    judge_model_args = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="judge_model_args",
        config_name="judge_model_args",
    )
    _append_json(command, "--judge-model-args", judge_model_args)
    judge_worker_num = pick(getattr(args, "judge_worker_num", None), settings.get("judge_worker_num"))
    if judge_worker_num is not None:
        command.extend(["--judge-worker-num", str(judge_worker_num)])
    sandbox = _config_json(
        args,
        root=root,
        settings=settings,
        arg_name="sandbox",
        config_name="sandbox",
    )
    _append_json(command, "--sandbox", sandbox)
    if not getattr(args, "no_agent_config", False):
        command.extend(
            [
                "--agent-config",
                json.dumps(_agent_config(args, root=root, config=config), ensure_ascii=False, separators=(",", ":")),
            ]
        )
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
    settings = table(config, "evalscope")
    mode = str(pick(getattr(args, "mode", None), settings.get("mode"), "native")).strip().lower()
    infer_args = infer_args_namespace(args, port=port_from_base_url(base_url))
    candidate_router = _parallel_candidate_enabled(args, settings)
    if mode == "native" and not candidate_router and getattr(args, "enable_auto_tool_choice", None) is None:
        # EvalScope Agent sends OpenAI tools.  Let vllm-rwkv render its native
        # RWKV tool template and expose the parser output as message.tool_calls.
        infer_args.enable_auto_tool_choice = True
    return build_infer_plan(
        infer_args,
        root=root,
        env=env,
        config=config,
    )


def _parallel_candidate_enabled(args: Any, settings: dict[str, Any]) -> bool:
    value = pick(
        getattr(args, "parallel_candidate_router", None),
        settings.get("parallel_candidate_router"),
        False,
    )
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    return bool(value)


def _parallel_candidate_config(args: Any, settings: dict[str, Any]) -> ParallelCandidateConfig:
    raw = settings.get("parallel_candidate_router")
    configured = raw if isinstance(raw, dict) else {}
    extra = settings.get("parallel_candidate_router_config")
    if isinstance(extra, dict):
        configured = {**configured, **extra}

    def positive(name: str, config_name: str, default: int) -> int:
        value = pick(getattr(args, name, None), configured.get(config_name), default)
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"parallel candidate {name} must be a positive integer") from error
        if result < 1:
            raise SystemExit(f"parallel candidate {name} must be a positive integer")
        return result

    def nonnegative(name: str, config_name: str, default: int) -> int:
        value = pick(getattr(args, name, None), configured.get(config_name), default)
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"parallel candidate {name} must be a non-negative integer") from error
        if result < 0:
            raise SystemExit(f"parallel candidate {name} must be a non-negative integer")
        return result

    defaults = ParallelCandidateConfig()
    fallback = pick(
        getattr(args, "parallel_candidate_fallback", None),
        configured.get("fallback_to_highest_confidence"),
        True,
    )
    return ParallelCandidateConfig(
        chunk_tools=positive("candidate_chunk_tools", "chunk_tools", 2),
        batch_size=positive("candidate_batch_size", "batch_size", 16),
        context_chars=positive("candidate_context_chars", "context_chars", 6000),
        prompt_max_chars=positive("candidate_prompt_max_chars", "prompt_max_chars", 12288),
        candidate_max_tokens=positive("candidate_max_tokens", "candidate_max_tokens", 2048),
        aggregate_max_tokens=positive("aggregate_max_tokens", "aggregate_max_tokens", 2048),
        max_candidates=positive("candidate_max_candidates", "max_candidates", 12),
        fallback_to_highest_confidence=bool(fallback),
        long_doc_min_chars=positive("long_doc_min_chars", "long_doc_min_chars", defaults.long_doc_min_chars),
        long_doc_max_chars=positive("long_doc_max_chars", "long_doc_max_chars", defaults.long_doc_max_chars),
        long_doc_overlap_lines=nonnegative(
            "long_doc_overlap_lines", "long_doc_overlap_lines", defaults.long_doc_overlap_lines
        ),
        long_doc_max_evidence_chunks=positive(
            "long_doc_max_evidence_chunks",
            "long_doc_max_evidence_chunks",
            defaults.long_doc_max_evidence_chunks,
        ),
        long_doc_max_evidence_chars=positive(
            "long_doc_max_evidence_chars",
            "long_doc_max_evidence_chars",
            defaults.long_doc_max_evidence_chars,
        ),
    )


def run_evalscope(args: Any, *, root: Path, env: dict[str, str], config: dict[str, Any]) -> int:
    if getattr(args, "report_only", False):
        output_dir = _output_dir(config, root=root, env=env, args=args)
        work_dir = _latest_evalscope_work_dir(output_dir)
        outer_trace = output_dir / "raw" / "trace_report.json"
        trace_report = work_dir / "raw" / "trace_report.json"
        if not trace_report.is_file():
            trace_report = outer_trace
        report = write_acceptance_report(
            work_dir,
            exit_code=_saved_trace_exit_code(trace_report),
            trace_report_path=trace_report if trace_report.is_file() else None,
        )
        print(f"evalscope: acceptance report written to {report}")
        if getattr(args, "scoreboard", False):
            _persist_evalscope_scoreboard(
                work_dir=work_dir,
                args=args,
                root=root,
                env=env,
                config=config,
            )
        return 0
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
            False,
        )
    )
    mode = str(pick(getattr(args, "mode", None), settings.get("mode"), "native")).strip().lower()
    use_candidate_proxy = _parallel_candidate_enabled(args, settings)
    if mode == "native" and use_naive_proxy:
        raise SystemExit(
            "native EvalScope Agent mode requires the OpenAI tools/tool_choice fields to reach "
            "vllm-rwkv; use --no-naive-chat-proxy (or set [evalscope].naive_chat_proxy = false)"
        )
    if use_candidate_proxy and mode != "native":
        raise SystemExit("parallel-candidate routing is only supported for native EvalScope Agent mode")
    if use_candidate_proxy and use_naive_proxy:
        raise SystemExit("parallel-candidate routing and naive Chat proxy are mutually exclusive")
    plan = build_evalscope_plan(args, root=root, env=env, config=config)
    infer_plan = _infer_plan(args, root=root, env=env, config=config, base_url=base_url)
    if getattr(args, "dry_run", False):
        if infer_plan is not None:
            print(format_plan_for_display(infer_plan))
        print(format_plan_for_display(plan))
        return 0

    server_process: subprocess.Popen[bytes] | None = None
    server_log: Path | None = None
    proxy: NaiveChatProxy | ParallelCandidateProxy | None = None
    run_exit_code: int | None = None
    trace_report_path: Path | None = None
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
            model = resolve_model_entry(config, args.model)
            api_key = pick(
                getattr(args, "api_key", None),
                model.get("api_key"),
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
        elif use_candidate_proxy:
            model = resolve_model_entry(config, args.model)
            api_key = pick(
                getattr(args, "api_key", None),
                model.get("api_key"),
                env_value(env, "HELICOPTER_EVAL_API_KEY", "OPENAI_API_KEY"),
                settings.get("api_key"),
                "EMPTY",
            )
            proxy = ParallelCandidateProxy(
                base_url,
                api_key=str(api_key),
                trace_path=output_dir / "raw" / "parallel_candidate.jsonl",
                config=_parallel_candidate_config(args, settings),
            )
            proxy.start()
            plan = build_evalscope_plan(
                args,
                root=root,
                env=env,
                config=config,
                api_url=proxy.base_url,
            )
            print(f"evalscope: parallel-candidate proxy listening at {proxy.base_url}; upstream {base_url}")
        child_env = _evalscope_child_env(plan.env, plan.command[plan.command.index("--api-url") + 1])
        run_exit_code = run_command(
            plan.command,
            cwd=plan.cwd,
            env=child_env,
            shown_env=plan.shown_env,
            dry_run=False,
        )
        return run_exit_code
    finally:
        if proxy is not None:
            proxy.close()
            trace_report = output_dir / "raw" / "trace_report.json"
            write_trace_report(proxy.trace_path, trace_report, exit_code=run_exit_code)
            print(f"evalscope: raw request/response trace report written to {trace_report}")
            work_dir = _latest_evalscope_work_dir(output_dir)
            trace_report_path = work_dir / "raw" / "trace_report.json"
            if trace_report_path != trace_report:
                write_trace_report(proxy.trace_path, trace_report_path, exit_code=run_exit_code)
                print(f"evalscope: run trace report written to {trace_report_path}")
        else:
            work_dir = _latest_evalscope_work_dir(output_dir)
        acceptance_report = write_acceptance_report(
            work_dir,
            exit_code=run_exit_code,
            trace_report_path=trace_report_path,
        )
        print(f"evalscope: acceptance report written to {acceptance_report}")
        if getattr(args, "scoreboard", False):
            _persist_evalscope_scoreboard(
                work_dir=work_dir,
                args=args,
                root=root,
                env=env,
                config=config,
            )
        if server_process is not None and not getattr(args, "keep_server", False):
            print("evalscope: stopping vLLM server")
            stop_server(server_process)
        elif server_process is not None:
            print(f"evalscope: leaving vLLM server running (pid {server_process.pid})")
