from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Literal, Mapping


EvaluationRoute = Literal["native", "rwkv"]


class RouteError(ValueError):
    pass


def read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as error:
        raise RouteError(f"lm-eval config not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise RouteError(f"invalid lm-eval TOML: {error}") from error


def select_route(config: Mapping[str, object]) -> EvaluationRoute:
    if "backend" not in config:
        return "native"
    if config["backend"] == "vllm_http":
        return "rwkv"
    raise RouteError(
        'backend must be "vllm_http" for RWKV or omitted for native lm-eval'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helicopter-lm-eval-route")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(select_route(read_toml(args.config)))
    except RouteError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
