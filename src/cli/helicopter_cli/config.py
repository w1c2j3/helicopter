from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .env import env_value, pick
from .paths import resolve_path


DEFAULT_EXAMPLE_CONFIG = Path("configs/example.toml")


def default_config_path(root: Path) -> Path:
    return root / DEFAULT_EXAMPLE_CONFIG


def table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(
    root: Path,
    config_path: str | None,
) -> tuple[dict[str, Any], Path]:
    path = Path(config_path) if config_path else default_config_path(root)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise SystemExit(f"config file not found: {path}")
    with path.open("rb") as file:
        config = tomllib.load(file)
    model_catalog = config.get("model_catalog", {})
    if isinstance(model_catalog, dict) and model_catalog.get("path"):
        merge_model_catalog(config, root=root, catalog_path=str(model_catalog["path"]))
    benchmark_config = config.get("benchmarks", {})
    if isinstance(benchmark_config, dict) and benchmark_config.get("index"):
        from .benchmark_specs import benchmark_specs_by_task, load_benchmark_index

        index_path = Path(str(benchmark_config["index"]))
        if not index_path.is_absolute():
            index_path = root / index_path
        specs = load_benchmark_index(index_path)
        config["_benchmark_specs"] = benchmark_specs_by_task(specs)
        config["_benchmark_index_path"] = str(index_path.resolve())
    elif isinstance(config.get("benchmark"), dict):
        from .benchmark_specs import benchmark_specs_by_task, load_benchmark_spec

        try:
            spec = load_benchmark_spec(path)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        config["_benchmark_specs"] = benchmark_specs_by_task([spec])
        config["_benchmark_config_path"] = str(path.resolve())
    return config, path


def merge_model_catalog(config: dict[str, Any], *, root: Path, catalog_path: str) -> None:
    """Merge a model inventory selected by the caller into a benchmark config."""

    path = Path(catalog_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise SystemExit(f"model catalog not found: {path}")
    with path.open("rb") as file:
        catalog = tomllib.load(file)
    catalog_models = catalog.get("models", {})
    if not isinstance(catalog_models, dict):
        raise SystemExit(f"[models] must be a TOML table in {path}")
    local_models = config.get("models", {})
    if not isinstance(local_models, dict):
        raise SystemExit("[models] must be a TOML table")
    config["models"] = {**catalog_models, **local_models}
    config["_model_catalog_path"] = str(path.resolve())
    config["_model_runtime"] = catalog.get("runtime", {})


def table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def resolve_model_entry(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = table(config, "models")
    seen: list[str] = []
    current_name = model_name
    while True:
        entry = models.get(current_name)
        if not isinstance(entry, dict):
            raise SystemExit(f"model alias not found in config: {model_name}")
        if current_name in seen:
            chain = " -> ".join([*seen, current_name])
            raise SystemExit(f"cyclic model alias in config: {chain}")
        seen.append(current_name)
        alias = entry.get("alias")
        if not alias:
            resolved = dict(entry)
            resolved.setdefault("name", current_name)
            resolved.setdefault("requested_name", model_name)
            return resolved
        current_name = str(alias)


def resolve_model_path(
    config: dict[str, Any],
    model_name: str,
    *,
    root: Path,
    env: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    entry = resolve_model_entry(config, model_name)
    paths = table(config, "paths")

    strict_checkpoint_path = env_value(env, "HELICOPTER_CHECKPOINT_PATH")
    if strict_checkpoint_path:
        return resolve_path(strict_checkpoint_path, root=root, env=env), entry
    if "path" in entry:
        return resolve_path(str(entry["path"]), root=root, env=env), entry

    filename = entry.get("file")
    if not filename:
        raise SystemExit(f"model {model_name} needs either path or file in config")
    base_value = pick(
        entry.get("weight_path"),
        paths.get("weight_path"),
        env_value(env, "WEIGHT_PATH", "HELICOPTER_WEIGHT_PATH"),
    )
    if not base_value:
        raise SystemExit(
            "WEIGHT_PATH is not set and config paths.weight_path is missing"
        )
    base = resolve_path(str(base_value), root=root, env=env)
    base_dir = base.parent if base.suffix == ".pth" else base
    return base_dir / str(filename), entry
