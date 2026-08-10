from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from helicopter_lighteval.http_pool import PoolError, PoolManifest

from .prompts import GENERATION_PROMPTS, PROMPT_PROFILES, get_prompt_profile


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
    "publish",
    "weights",
    "wkv_modes",
    "pool_manifests",
    "prompt",
    "generation_kwargs",
    "task_include_paths",
    "benchmark_configs",
}
_PROMPT_FIELDS = {
    "profile",
    "generation_prompt",
    "system_instruction",
    "num_fewshot",
    "fewshot_as_multiturn",
}
_GENERATION_FIELDS = {
    "do_sample",
    "max_gen_toks",
    "max_new_tokens",
    "min_p",
    "num_beams",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "until",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "penalty_decay",
    "ignore_eos",
}
_BENCHMARK_FIELDS = {
    "schema_version",
    "selector",
    "batch_size",
    "max_gen_toks",
    "limit",
    "confirm_run_unsafe_code",
    "trust_remote_dataset_code",
    "dataset_path_override",
    "dataset_kwargs_override",
    "prompt",
    "generation_kwargs",
}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionUnit:
    weight: Path | None
    weight_sha256: str | None
    wkv_mode: str
    manifest_path: Path
    manifest: PoolManifest


@dataclass(frozen=True)
class PromptConfig:
    profile: str = "none"
    generation_prompt: str = "none"
    system_instruction: str | None = None
    num_fewshot: int | None = None
    fewshot_as_multiturn: bool = True

    @property
    def apply_chat_template(self) -> bool:
        return self.profile != "none"

    @property
    def stop(self) -> str | None:
        return get_prompt_profile(self.profile).stop

    def public(self) -> dict[str, object]:
        profile = get_prompt_profile(self.profile)
        instruction_sha256 = (
            hashlib.sha256(self.system_instruction.encode("utf-8")).hexdigest()
            if self.system_instruction is not None
            else None
        )
        return {
            "profile": self.profile,
            "profile_sha256": profile.sha256,
            "generation_prompt": self.generation_prompt,
            "system_instruction": self.system_instruction,
            "system_instruction_sha256": instruction_sha256,
            "num_fewshot": self.num_fewshot,
            "fewshot_as_multiturn": self.fewshot_as_multiturn,
            "stop": profile.stop,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    selector: str
    path: Path
    batch_size: int
    max_gen_toks: int
    limit: int | None
    confirm_run_unsafe_code: bool
    trust_remote_dataset_code: bool
    dataset_path_override: str | None
    dataset_kwargs_override: dict[str, object] | None
    prompt: PromptConfig
    generation_kwargs: dict[str, object] = field(default_factory=dict)

    def public(self) -> dict[str, object]:
        return {
            "selector": self.selector,
            "path": str(self.path),
            "batch_size": self.batch_size,
            "max_gen_toks": self.max_gen_toks,
            "limit": self.limit,
            "confirm_run_unsafe_code": self.confirm_run_unsafe_code,
            "trust_remote_dataset_code": self.trust_remote_dataset_code,
            "dataset_path_override": self.dataset_path_override,
            "dataset_kwargs_override": self.dataset_kwargs_override,
            "prompt": self.prompt.public(),
            "generation_kwargs": dict(self.generation_kwargs),
        }


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
    publish: bool = False
    execution_units: tuple[ExecutionUnit, ...] = ()
    scoreboard_url: str | None = None
    scoreboard_token: str | None = None
    backend: str = "vllm_http"
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation_kwargs: dict[str, object] = field(default_factory=dict)
    task_include_paths: tuple[Path, ...] = ()
    benchmarks: tuple[BenchmarkConfig, ...] = ()

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
        log_samples = raw.get("log_samples", True)
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

        publish = raw.get("publish", False)
        if not isinstance(publish, bool):
            raise ConfigError("publish must be a boolean")
        if publish and limit is not None:
            raise ConfigError("published evaluation must not set limit")

        prompt = cls._prompt(raw.get("prompt"))
        generation_kwargs = cls._generation_kwargs(raw.get("generation_kwargs"))
        configured_benchmark_paths = raw.get("benchmark_configs")
        benchmarks = (
            tuple(
                cls._benchmark_config(
                    value,
                    path.parent,
                    prompt=prompt,
                    generation_kwargs=generation_kwargs,
                    batch_size=batch_size,
                    max_gen_toks=max_gen_toks,
                    limit=limit,
                )
                for value in cls._strings(
                    configured_benchmark_paths, "benchmark_configs"
                )
            )
            if configured_benchmark_paths is not None
            else ()
        )
        if benchmarks:
            selectors = tuple(benchmark.selector for benchmark in benchmarks)
            missing_benchmarks = sorted(set(tasks) - set(selectors))
            extra_benchmarks = sorted(set(selectors) - set(tasks))
            duplicate_benchmarks = sorted(
                selector
                for selector in set(selectors)
                if selectors.count(selector) > 1
            )
            if duplicate_benchmarks:
                raise ConfigError(
                    "duplicate benchmark config selectors: "
                    + ", ".join(duplicate_benchmarks)
                )
            if missing_benchmarks or extra_benchmarks:
                details = []
                if missing_benchmarks:
                    details.append("missing " + ", ".join(missing_benchmarks))
                if extra_benchmarks:
                    details.append("unexpected " + ", ".join(extra_benchmarks))
                raise ConfigError(
                    "benchmark configs must match tasks exactly: " + "; ".join(details)
                )
        configured_task_paths = raw.get("task_include_paths")
        task_include_paths = (
            tuple(
                cls._task_include_path(
                    cls._expand_environment(value, env, "task_include_paths"),
                    path.parent,
                )
                for value in cls._strings(
                    configured_task_paths, "task_include_paths"
                )
            )
            if configured_task_paths is not None
            else ()
        )

        configured_manifests = raw.get("pool_manifests")
        if configured_manifests is None:
            configured_manifest = env.get("HELICOPTER_VLLM_POOL_MANIFEST")
            if not configured_manifest:
                raise ConfigError(
                    "backend = vllm_http requires pool_manifests or "
                    "HELICOPTER_VLLM_POOL_MANIFEST"
                )
            manifest_values = (configured_manifest,)
        else:
            manifest_values = cls._strings(configured_manifests, "pool_manifests")
        manifest_paths = tuple(
            cls._manifest_path(cls._expand_environment(value, env, "pool_manifests"))
            for value in manifest_values
        )
        manifests: list[PoolManifest] = []
        for manifest_path in manifest_paths:
            try:
                manifest = PoolManifest.read(manifest_path)
            except PoolError as error:
                raise ConfigError(str(error)) from error
            if manifest.max_model_len < 4:
                raise ConfigError("vLLM pool max_model_len must be at least 4 for scoring")
            if max_gen_toks >= manifest.max_model_len - 2:
                raise ConfigError(
                    "max_gen_toks must be smaller than the effective vLLM context length"
                )
            manifests.append(manifest)
        for benchmark in benchmarks:
            if benchmark.max_gen_toks >= min(
                manifest.max_model_len - 2 for manifest in manifests
            ):
                raise ConfigError(
                    f"benchmark {benchmark.selector} max_gen_toks must be smaller "
                    "than the effective vLLM context length"
                )
            if publish and benchmark.limit is not None:
                raise ConfigError(
                    f"published benchmark {benchmark.selector} must not set limit"
                )

        configured_weights = raw.get("weights")
        weights: tuple[tuple[Path, str], ...] = ()
        if configured_weights is not None:
            weight_values = cls._strings(configured_weights, "weights")
            weight_root_raw = env.get("WEIGHT_PATH")
            if not weight_root_raw:
                raise ConfigError("weights require WEIGHT_PATH")
            weight_root = Path(weight_root_raw).expanduser().resolve(strict=True)
            weights = tuple(
                cls._weight(
                    cls._expand_environment(value, env, "weights"), weight_root
                )
                for value in weight_values
            )
            hashes = [digest for _path, digest in weights]
            if len(hashes) != len(set(hashes)):
                raise ConfigError("duplicate weight content is not allowed")

        configured_modes = raw.get("wkv_modes")
        if configured_modes is None:
            wkv_modes = tuple(dict.fromkeys(item.wkv_mode for item in manifests))
        else:
            wkv_modes = cls._strings(configured_modes, "wkv_modes")
        if any(mode not in {"fp16", "fp32io16"} for mode in wkv_modes):
            raise ConfigError("wkv_modes contains an unsupported mode")

        units: list[ExecutionUnit] = []
        if weights:
            dimensions = [
                (weight, digest, mode)
                for weight, digest in weights
                for mode in wkv_modes
            ]
            if len(dimensions) != len(manifests):
                raise ConfigError(
                    "pool_manifests must contain one entry per weight and WKV mode"
                )
            for (weight, digest, mode), manifest_path, manifest in zip(
                dimensions, manifest_paths, manifests, strict=True
            ):
                if manifest.wkv_mode != mode:
                    raise ConfigError("pool manifest WKV mode does not match its matrix unit")
                if manifest.weight_sha256 != digest:
                    raise ConfigError(
                        "pool manifest weight_sha256 does not match its matrix unit"
                    )
                if manifest.weight_display_name != weight.name:
                    raise ConfigError(
                        "pool manifest weight_display_name does not match its matrix unit"
                    )
                units.append(
                    ExecutionUnit(weight, digest, mode, manifest_path, manifest)
                )
        else:
            if len(manifests) != 1:
                raise ConfigError("multiple pool manifests require configured weights")
            units.append(
                ExecutionUnit(
                    None,
                    manifests[0].weight_sha256,
                    manifests[0].wkv_mode,
                    manifest_paths[0],
                    manifests[0],
                )
            )
        if publish and (not weights or set(wkv_modes) != {"fp16", "fp32io16"}):
            raise ConfigError(
                "published evaluation requires weights and both WKV modes"
            )

        scoreboard_url: str | None = None
        scoreboard_token: str | None = None
        if publish:
            scoreboard_url = env.get("HELICOPTER_SCOREBOARD_URL")
            scoreboard_token = env.get("HELICOPTER_SCOREBOARD_TOKEN")
            if not scoreboard_url or not scoreboard_token:
                raise ConfigError(
                    "published evaluation requires Scoreboard URL and token"
                )
            parsed = urlsplit(scoreboard_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ConfigError("HELICOPTER_SCOREBOARD_URL must be an HTTP(S) base URL")
            scoreboard_url = scoreboard_url.rstrip("/")

        vllm_pool_manifest = manifest_paths[0]
        manifest = manifests[0]

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
            publish=publish,
            execution_units=tuple(units),
            scoreboard_url=scoreboard_url,
            scoreboard_token=scoreboard_token,
            prompt=prompt,
            generation_kwargs=generation_kwargs,
            task_include_paths=task_include_paths,
            benchmarks=benchmarks,
        )

    @staticmethod
    def _prompt(raw: object, base: PromptConfig | None = None) -> PromptConfig:
        if raw is None:
            return base or PromptConfig()
        if not isinstance(raw, dict):
            raise ConfigError("prompt must be a TOML table")
        unknown = sorted(set(raw) - _PROMPT_FIELDS)
        if unknown:
            raise ConfigError("unknown prompt fields: " + ", ".join(unknown))
        profile = raw.get("profile", base.profile if base is not None else "none")
        if not isinstance(profile, str) or profile not in PROMPT_PROFILES:
            raise ConfigError(
                "prompt.profile must be one of: " + ", ".join(PROMPT_PROFILES)
            )
        default_generation = (
            base.generation_prompt
            if base is not None and profile == base.profile
            else "none"
        )
        generation_prompt = raw.get("generation_prompt", default_generation)
        if (
            not isinstance(generation_prompt, str)
            or generation_prompt not in GENERATION_PROMPTS
        ):
            raise ConfigError(
                "prompt.generation_prompt must be one of: "
                + ", ".join(GENERATION_PROMPTS)
            )
        if profile == "none" and generation_prompt != "none":
            raise ConfigError(
                "prompt.generation_prompt requires an enabled RWKV prompt profile"
            )
        system_instruction = raw.get(
            "system_instruction", base.system_instruction if base is not None else None
        )
        if system_instruction is not None and (
            not isinstance(system_instruction, str)
            or not system_instruction.strip()
            or system_instruction != system_instruction.strip()
        ):
            raise ConfigError(
                "prompt.system_instruction must be non-empty trimmed text"
            )
        num_fewshot = raw.get(
            "num_fewshot", base.num_fewshot if base is not None else None
        )
        if num_fewshot is not None and (
            not isinstance(num_fewshot, int)
            or isinstance(num_fewshot, bool)
            or num_fewshot < 0
        ):
            raise ConfigError("prompt.num_fewshot must be a non-negative integer")
        fewshot_as_multiturn = raw.get(
            "fewshot_as_multiturn",
            base.fewshot_as_multiturn if base is not None else True,
        )
        if not isinstance(fewshot_as_multiturn, bool):
            raise ConfigError("prompt.fewshot_as_multiturn must be a boolean")
        return PromptConfig(
            profile=profile,
            generation_prompt=generation_prompt,
            system_instruction=system_instruction,
            num_fewshot=num_fewshot,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    @staticmethod
    def _generation_kwargs(raw: object) -> dict[str, object]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError("generation_kwargs must be a TOML table")
        unknown = sorted(set(raw) - _GENERATION_FIELDS)
        if unknown:
            raise ConfigError(
                "unknown generation_kwargs fields: " + ", ".join(unknown)
            )
        if "max_gen_toks" in raw and "max_new_tokens" in raw:
            raise ConfigError(
                "generation_kwargs cannot set both max_gen_toks and max_new_tokens"
            )
        for name in ("max_gen_toks", "max_new_tokens", "top_k", "seed"):
            value = raw.get(name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or (name != "seed" and value <= 0)
            ):
                raise ConfigError(f"generation_kwargs.{name} must be an integer")
        for name in ("do_sample", "ignore_eos"):
            value = raw.get(name)
            if value is not None and not isinstance(value, bool):
                raise ConfigError(f"generation_kwargs.{name} must be a boolean")
        for name in (
            "temperature",
            "top_p",
            "min_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "penalty_decay",
        ):
            value = raw.get(name)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise ConfigError(f"generation_kwargs.{name} must be numeric")
        num_beams = raw.get("num_beams")
        if num_beams is not None and num_beams != 1:
            raise ConfigError("generation_kwargs.num_beams must be 1")
        until = raw.get("until")
        if until is not None and not (
            isinstance(until, str)
            and until
            or isinstance(until, list)
            and until
            and all(isinstance(value, str) and value for value in until)
        ):
            raise ConfigError(
                "generation_kwargs.until must be non-empty text or an array of text"
            )
        return dict(raw)

    @classmethod
    def _benchmark_config(
        cls,
        raw_path: str,
        config_root: Path,
        *,
        prompt: PromptConfig,
        generation_kwargs: dict[str, object],
        batch_size: int,
        max_gen_toks: int,
        limit: int | None,
    ) -> BenchmarkConfig:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config_root / path
        if path.is_symlink():
            raise ConfigError("benchmark config paths must not contain symlinks")
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ConfigError(f"benchmark config is missing: {path}") from error
        if not path.is_file():
            raise ConfigError("benchmark config path must be a regular file")
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"invalid benchmark TOML {path}: {error}") from error
        unknown = sorted(set(raw) - _BENCHMARK_FIELDS)
        if unknown:
            raise ConfigError(
                f"unknown benchmark config fields in {path}: " + ", ".join(unknown)
            )
        missing = sorted({"schema_version", "selector"} - set(raw))
        if missing:
            raise ConfigError(
                f"missing benchmark config fields in {path}: " + ", ".join(missing)
            )
        if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
            raise ConfigError(f"benchmark schema_version must be 1 in {path}")
        selector = raw["selector"]
        if (
            not isinstance(selector, str)
            or not selector
            or selector != selector.strip()
        ):
            raise ConfigError(f"benchmark selector must be non-empty trimmed text in {path}")
        configured_batch_size = raw.get("batch_size", batch_size)
        if (
            not isinstance(configured_batch_size, int)
            or isinstance(configured_batch_size, bool)
            or configured_batch_size <= 0
        ):
            raise ConfigError(f"benchmark batch_size must be a positive integer in {path}")
        configured_max_gen_toks = raw.get("max_gen_toks", max_gen_toks)
        if (
            not isinstance(configured_max_gen_toks, int)
            or isinstance(configured_max_gen_toks, bool)
            or configured_max_gen_toks <= 0
        ):
            raise ConfigError(
                f"benchmark max_gen_toks must be a positive integer in {path}"
            )
        configured_limit = raw.get("limit", limit)
        if configured_limit is not None and (
            not isinstance(configured_limit, int)
            or isinstance(configured_limit, bool)
            or configured_limit <= 0
        ):
            raise ConfigError(f"benchmark limit must be a positive integer in {path}")
        confirm_run_unsafe_code = raw.get("confirm_run_unsafe_code", False)
        if not isinstance(confirm_run_unsafe_code, bool):
            raise ConfigError(
                f"benchmark confirm_run_unsafe_code must be a boolean in {path}"
            )
        trust_remote_dataset_code = raw.get("trust_remote_dataset_code", False)
        if not isinstance(trust_remote_dataset_code, bool):
            raise ConfigError(
                f"benchmark trust_remote_dataset_code must be a boolean in {path}"
            )
        dataset_path_override = raw.get("dataset_path_override")
        if dataset_path_override is not None and (
            not isinstance(dataset_path_override, str)
            or dataset_path_override != dataset_path_override.strip()
            or dataset_path_override.count("/") != 1
            or any(not part for part in dataset_path_override.split("/"))
        ):
            raise ConfigError(
                f"benchmark dataset_path_override must be namespace/name in {path}"
            )
        dataset_kwargs_override = raw.get("dataset_kwargs_override")
        if dataset_kwargs_override is not None and not isinstance(
            dataset_kwargs_override, dict
        ):
            raise ConfigError(
                f"benchmark dataset_kwargs_override must be a table in {path}"
            )
        configured_prompt = cls._prompt(raw.get("prompt"), prompt)
        configured_generation = dict(generation_kwargs)
        configured_generation.update(
            cls._generation_kwargs(raw.get("generation_kwargs"))
        )
        return BenchmarkConfig(
            selector=selector,
            path=path,
            batch_size=configured_batch_size,
            max_gen_toks=configured_max_gen_toks,
            limit=configured_limit,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            trust_remote_dataset_code=trust_remote_dataset_code,
            dataset_path_override=dataset_path_override,
            dataset_kwargs_override=(
                dict(dataset_kwargs_override)
                if dataset_kwargs_override is not None
                else None
            ),
            prompt=configured_prompt,
            generation_kwargs=configured_generation,
        )

    @staticmethod
    def _task_include_path(raw: str, config_root: Path) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = config_root / path
        if path.is_symlink():
            raise ConfigError("task_include_paths must not contain symlinks")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ConfigError(f"task include path is missing: {path}") from error
        if not resolved.is_dir():
            raise ConfigError("task_include_paths must contain directories")
        return resolved

    @staticmethod
    def _manifest_path(raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ConfigError("pool manifest paths must be absolute")
        if not path.is_file() or path.is_symlink():
            raise ConfigError("pool manifest path must be a regular file")
        return path

    @staticmethod
    def _weight(raw: str, weight_root: Path) -> tuple[Path, str]:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"weight path must stay below WEIGHT_PATH: {raw}")
        candidate = weight_root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ConfigError(f"weight path must not use symlinks: {raw}")
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(weight_root)
        except (FileNotFoundError, ValueError) as error:
            raise ConfigError(f"weight path is missing or outside WEIGHT_PATH: {raw}") from error
        if not candidate.is_file():
            raise ConfigError(f"weight path is not a regular file: {raw}")
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return candidate, digest

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
            "publish": self.publish,
            "execution_units": [
                {
                    "weight": unit.weight.name if unit.weight else None,
                    "weight_sha256": unit.weight_sha256,
                    "wkv_mode": unit.wkv_mode,
                    "pool_manifest": str(unit.manifest_path),
                }
                for unit in self.execution_units
            ],
            "scoreboard_url": self.scoreboard_url,
            "scoreboard_token": "[REDACTED]" if self.scoreboard_token else None,
            "prompt": self.prompt.public(),
            "generation_kwargs": dict(self.generation_kwargs),
            "task_include_paths": [str(path) for path in self.task_include_paths],
            "benchmark_configs": [
                benchmark.public() for benchmark in self.benchmarks
            ],
        }

    def benchmark_for_selector(self, selector: str) -> BenchmarkConfig | None:
        return next(
            (
                benchmark
                for benchmark in self.benchmarks
                if benchmark.selector == selector
            ),
            None,
        )
