from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from helicopter_lighteval.http_pool import PoolError, PoolManifest


_CONFIG_FIELDS = {
    "schema_version",
    "backend",
    "tasks",
    "output_dir",
    "batch_size",
    "eot_token_id",
    "max_gen_toks",
    "limit",
    "log_samples",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LMEvalConfig:
    tasks: tuple[str, ...]
    output_dir: Path
    batch_size: int
    eot_token_id: int
    max_gen_toks: int
    limit: int | None
    log_samples: bool
    vllm_pool_manifest: Path
    manifest: PoolManifest
    backend: str = "vllm_http"

    @classmethod
    def read(
        cls,
        path: Path,
        env: Mapping[str, str] = os.environ,
    ) -> LMEvalConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except FileNotFoundError as error:
            raise ConfigError(f"lm-eval config not found: {path}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid lm-eval TOML: {error}") from error

        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ConfigError("unknown lm-eval config fields: " + ", ".join(unknown))
        missing = sorted(
            {"schema_version", "backend", "tasks", "output_dir"} - set(raw)
        )
        if missing:
            raise ConfigError("missing lm-eval config fields: " + ", ".join(missing))
        if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
            raise ConfigError("schema_version must be 1")
        if raw["backend"] != "vllm_http":
            raise ConfigError("backend must be vllm_http")

        tasks = cls._strings(raw["tasks"], "tasks")
        configured_output = raw["output_dir"]
        if (
            not isinstance(configured_output, str)
            or not configured_output
            or configured_output != configured_output.strip()
        ):
            raise ConfigError("output_dir must be a non-empty trimmed string")
        configured_output = cls._expand_environment(
            configured_output, env, "output_dir"
        )
        output_dir = Path(configured_output).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        if output_dir.exists() and (
            not output_dir.is_dir() or output_dir.is_symlink()
        ):
            raise ConfigError("output_dir must be a regular directory")

        batch_size = raw.get("batch_size", 1)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ConfigError("batch_size must be a positive integer")
        eot_token_id = raw.get("eot_token_id", 0)
        if (
            not isinstance(eot_token_id, int)
            or isinstance(eot_token_id, bool)
            or eot_token_id < 0
        ):
            raise ConfigError("eot_token_id must be a non-negative integer")
        log_samples = raw.get("log_samples", False)
        if not isinstance(log_samples, bool):
            raise ConfigError("log_samples must be a boolean")
        max_gen_toks = raw.get("max_gen_toks", 256)
        if (
            not isinstance(max_gen_toks, int)
            or isinstance(max_gen_toks, bool)
            or max_gen_toks <= 0
        ):
            raise ConfigError("max_gen_toks must be a positive integer")
        limit = raw.get("limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ConfigError("limit must be a positive integer")

        configured_manifest = env.get("HELICOPTER_VLLM_POOL_MANIFEST")
        if not configured_manifest:
            raise ConfigError(
                "backend = vllm_http requires HELICOPTER_VLLM_POOL_MANIFEST"
            )
        vllm_pool_manifest = Path(configured_manifest).expanduser()
        if not vllm_pool_manifest.is_absolute():
            raise ConfigError("HELICOPTER_VLLM_POOL_MANIFEST must be an absolute path")
        if not vllm_pool_manifest.is_file() or vllm_pool_manifest.is_symlink():
            raise ConfigError(
                "HELICOPTER_VLLM_POOL_MANIFEST must be a regular file"
            )
        try:
            manifest = PoolManifest.read(vllm_pool_manifest)
        except PoolError as error:
            raise ConfigError(str(error)) from error
        if manifest.max_model_len < 4:
            raise ConfigError("vLLM pool max_model_len must be at least 4 for scoring")
        if max_gen_toks >= manifest.max_model_len - 2:
            raise ConfigError(
                "max_gen_toks must be smaller than the effective vLLM context length"
            )

        return cls(
            tasks=tasks,
            output_dir=output_dir,
            batch_size=batch_size,
            eot_token_id=eot_token_id,
            max_gen_toks=max_gen_toks,
            limit=limit,
            log_samples=log_samples,
            vllm_pool_manifest=vllm_pool_manifest,
            manifest=manifest,
        )

    @staticmethod
    def _strings(value: object, name: str) -> tuple[str, ...]:
        if (
            not isinstance(value, list)
            or not value
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in value
            )
        ):
            raise ConfigError(
                f"{name} must be a non-empty array of non-empty trimmed strings"
            )
        values = tuple(value)
        duplicates = sorted(item for item in set(values) if values.count(item) > 1)
        if duplicates:
            raise ConfigError(f"duplicate {name}: " + ", ".join(duplicates))
        return values

    @staticmethod
    def _expand_environment(
        value: str,
        env: Mapping[str, str],
        name: str,
    ) -> str:
        match = _ENV_REFERENCE.fullmatch(value)
        if match is None:
            return value
        variable = match.group(1)
        expanded = env.get(variable)
        if not expanded:
            raise ConfigError(
                f"{name} references missing environment variable {variable}"
            )
        return expanded

    def public(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "backend": self.backend,
            "tasks": list(self.tasks),
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "eot_token_id": self.eot_token_id,
            "max_gen_toks": self.max_gen_toks,
            "limit": self.limit,
            "log_samples": self.log_samples,
            "vllm_pool_manifest": str(self.vllm_pool_manifest),
        }
