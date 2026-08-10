from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from .commands import (
    EMB_DEVICES,
    WKV_MODES,
    build_infer_plan,
    build_takeoff_plan,
    prepend_venv_path,
)
from .config import load_config
from .env import DEFAULT_ENV_FILE, find_env_path, load_env
from .paths import find_root
from .runner import run_command


LOCAL_POOL_MANIFEST = Path(".tmp/runtime/rwkv-vllm-pool.json")


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file", default=DEFAULT_ENV_FILE, help="dotenv file to load first"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the command without executing it"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicopter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="start vLLM for an RWKV model")
    add_runtime_options(infer)
    infer.add_argument(
        "--config",
        help="serving TOML; defaults to configs/example.toml",
    )
    infer.add_argument("model", help="model alias from configs")
    infer.add_argument("--wkv-mode", choices=WKV_MODES)
    infer.add_argument("--emb-device", choices=EMB_DEVICES)
    infer.add_argument(
        "--allow-fp16-accumulation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    infer.add_argument("--host")
    infer.add_argument("--port")
    infer.add_argument("--served-model-name")
    infer.add_argument("--tensor-parallel-size", type=int)
    infer.add_argument("--gpu-memory-utilization", type=float)
    infer.add_argument("--max-model-len", type=int)
    infer.add_argument("--max-num-seqs", type=int)
    infer.add_argument("--max-num-batched-tokens", type=int)
    infer.add_argument("--enable-auto-tool-choice", action="store_true", default=None)
    infer.set_defaults(plan_builder=build_infer_plan)

    takeoff = subparsers.add_parser(
        "takeoff", help="launch a MaxRL experiment through Verl"
    )
    add_runtime_options(takeoff)
    takeoff.add_argument("--config", required=True, help="MaxRL experiment TOML")
    takeoff.add_argument(
        "--override", action="append", help="operational override validated by Verl"
    )
    takeoff.set_defaults(plan_builder=build_takeoff_plan)

    evaluate = subparsers.add_parser(
        "eval",
        help="run configured evaluation benchmarks",
    )
    evaluate.add_argument(
        "--evaluator",
        choices=("lighteval", "lm-eval"),
        default="lighteval",
        help="evaluation harness; defaults to lighteval",
    )
    evaluate.add_argument("--config", required=True, help="evaluation TOML")
    evaluate.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="private dotenv file; defaults to .env.local",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and print evaluator readiness",
    )

    publish = subparsers.add_parser(
        "publish",
        help="publish completed evaluation artifacts without rerunning a model",
    )
    publish.add_argument(
        "--evaluator",
        choices=("lm-eval",),
        default="lm-eval",
        help="artifact evaluator; defaults to lm-eval",
    )
    publish.add_argument(
        "--output-dir",
        action="append",
        required=True,
        help="completed evaluator output directory; repeat for matrix units",
    )
    publish.add_argument(
        "--weight-sha256",
        help="explicit weight SHA-256 when existing summary metadata lacks it",
    )
    publish.add_argument("--weight-display-name")
    publish.add_argument("--vllm-version", default="not-recorded-in-artifact")
    publish.add_argument("--torch-version", default="not-recorded-in-artifact")
    publish.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="private dotenv file; defaults to .env.local",
    )
    publish.add_argument(
        "--dry-run",
        action="store_true",
        help="validate artifacts and Scoreboard readiness without publishing",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = find_root()
    if args.command in {"eval", "publish"}:
        env_path = find_env_path(root, args.env_file, use_fallbacks=False)
        if env_path is not None:
            try:
                env_status = env_path.lstat()
            except OSError as error:
                parser.error(
                    f"cannot inspect evaluation private environment file: {error}"
                )
            if (
                not stat.S_ISREG(env_status.st_mode)
                or stat.S_IMODE(env_status.st_mode) != 0o600
                or env_status.st_uid != os.geteuid()
            ):
                parser.error(
                    "evaluation private environment file must be owned by the current "
                    "user, have mode 0600, and be a regular non-symlink file: "
                    f"{env_path}"
                )
        try:
            eval_env, _ = load_env(
                root,
                args.env_file,
                use_fallbacks=False,
                require_private=True,
            )
        except (OSError, UnicodeError) as error:
            parser.error(
                f"cannot securely read evaluation private environment file: {error}"
            )
        if args.command == "eval" and not eval_env.get(
            "HELICOPTER_VLLM_POOL_MANIFEST"
        ):
            local_manifest = root / LOCAL_POOL_MANIFEST
            if local_manifest.is_file() and not local_manifest.is_symlink():
                eval_env["HELICOPTER_VLLM_POOL_MANIFEST"] = str(local_manifest)
        is_lm_eval = args.evaluator == "lm-eval"
        python_variable = (
            "HELICOPTER_LM_EVAL_PYTHON"
            if is_lm_eval
            else "HELICOPTER_EVAL_PYTHON"
        )
        configured_python = eval_env.get(python_variable)
        eval_python = (
            Path(configured_python).expanduser()
            if configured_python
            else root
            / (
                ".venv-lm-eval/bin/python"
                if is_lm_eval
                else ".venv-lighteval/bin/python"
            )
        )
        if not eval_python.is_absolute():
            eval_python = root / eval_python
        if not os.access(eval_python, os.X_OK):
            evaluator_name = "lm-eval" if is_lm_eval else "LightEval"
            component = "lm-eval" if is_lm_eval else "lighteval"
            parser.error(
                f"{evaluator_name} Python executable not found: {eval_python}; "
                f"prepare the {component} component"
            )
        if args.command == "eval":
            config_path = Path(args.config).expanduser()
            if not config_path.is_absolute():
                config_path = Path.cwd() / config_path
            command = [
                str(eval_python),
                "-m",
                "helicopter_lm_eval" if is_lm_eval else "helicopter_lighteval",
                "--config",
                str(config_path),
            ]
        else:
            command = [
                str(eval_python),
                "-m",
                "helicopter_lm_eval",
                "--publish-existing",
            ]
            for output_dir in args.output_dir:
                command.extend(["--output-dir", str(Path(output_dir).expanduser())])
            for option in (
                "weight_sha256",
                "weight_display_name",
                "vllm_version",
                "torch_version",
            ):
                value = getattr(args, option)
                if value:
                    command.extend(["--" + option.replace("_", "-"), str(value)])
        if args.dry_run:
            command.append("--dry-run")
        return run_command(
            command,
            cwd=root,
            env=eval_env,
            shown_env={},
            dry_run=False,
        )

    env, _ = load_env(root, args.env_file)
    prepend_venv_path(env, root)

    if args.command == "takeoff":
        plan = args.plan_builder(args, root=root, env=env)
    else:
        config, _ = load_config(root, args.config)
        plan = args.plan_builder(args, root=root, env=env, config=config)
    return run_command(
        plan.command,
        cwd=plan.cwd,
        env=plan.env,
        shown_env=plan.shown_env,
        dry_run=args.dry_run and args.command != "takeoff",
    )


if __name__ == "__main__":
    raise SystemExit(main())
