from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

from helicopter_lighteval.publish import PublicationError

from .config import ConfigError
from .existing import publish_existing
from .native import NativeConfigError, run as run_native
from .route import RouteError, read_toml, select_route


def run_rwkv(*, config_path: Path, env: Mapping[str, str], dry_run: bool) -> int:
    from .evaluate import run

    return run(config_path=config_path, env=env, dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helicopter-lm-eval")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--publish-existing",
        action="store_true",
        help="publish completed lm-eval output directories without rerunning the model",
    )
    parser.add_argument(
        "--output-dir",
        action="append",
        type=Path,
        help="completed lm-eval output directory (repeat for matrix units)",
    )
    parser.add_argument(
        "--weight-sha256",
        help="explicit weight SHA-256 for artifacts whose summary lacks it",
    )
    parser.add_argument("--weight-display-name")
    parser.add_argument(
        "--vllm-version",
        default="not-recorded-in-artifact",
        help="runtime version to record when existing artifacts lack it",
    )
    parser.add_argument(
        "--torch-version",
        default="not-recorded-in-artifact",
        help="runtime version to record when existing artifacts lack it",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.publish_existing:
            if args.config is not None:
                parser.error("--publish-existing cannot be combined with --config")
            if not args.output_dir:
                parser.error("--publish-existing requires at least one --output-dir")
            return publish_existing(
                output_dirs=args.output_dir,
                env=os.environ,
                dry_run=args.dry_run,
                weight_sha256=args.weight_sha256,
                weight_display_name=args.weight_display_name,
                vllm_version=args.vllm_version,
                torch_version=args.torch_version,
            )
        if (
            args.output_dir
            or args.weight_sha256
            or args.weight_display_name
            or args.vllm_version != "not-recorded-in-artifact"
            or args.torch_version != "not-recorded-in-artifact"
        ):
            parser.error("publication options require --publish-existing")
        if args.config is None:
            parser.error("the following arguments are required: --config")
        raw_config = read_toml(args.config)
        route = select_route(raw_config)
        if route == "rwkv":
            return run_rwkv(
                config_path=args.config,
                env=os.environ,
                dry_run=args.dry_run,
            )
        return run_native(config=raw_config, dry_run=args.dry_run)
    except (ConfigError, NativeConfigError, PublicationError, RouteError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
