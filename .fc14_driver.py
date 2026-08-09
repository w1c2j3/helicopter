import importlib.util
import json
from pathlib import Path


SOURCE = Path("/home/rwkv/chase/eval-results/.fc12_pipeline.py")
spec = importlib.util.spec_from_file_location("fc12_pipeline", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load {SOURCE}")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

pipeline.OUTROOT = Path("/home/rwkv/chase/eval-results/evalscope-g1h-20260805-fc14")
pipeline.MODELS = [
    ("72", "g1h-7.2b", 29572, 32),
    ("133", "g1h-13.3b", 29533, 16),
    ("29", "g1h-2.9b", 19429, 128),
    ("15", "g1h-1.5b", 19415, 128),
]

pipeline.ENV["TAU2_DATA_DIR"] = (
    "/home/rwkv/chase/rwkv-skills-g1h-normal-20260719/assets/agent_bench/tau_v2/data"
)
pipeline.ENV["LD_LIBRARY_PATH"] = ":".join(
    [
        "/home/rwkv/chase/eval-results/portaudio-local/lib",
        pipeline.ENV.get("LD_LIBRARY_PATH", ""),
    ]
)


def main():
    pipeline.OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += pipeline.run_stage(
        "core",
        pipeline.CORE,
        {"72": 32, "133": 16, "29": 128, "15": 128},
    )
    rows += pipeline.run_stage(
        "mcp",
        pipeline.MCP,
        {"72": 32, "133": 16, "29": 64, "15": 64},
    )
    rows += pipeline.run_stage(
        "skillsbench",
        pipeline.SKILLSBENCH,
        {"72": 8, "133": 4, "29": 8, "15": 8},
    )
    rows += pipeline.run_stage(
        "general",
        pipeline.GENERAL,
        {"72": 32, "133": 16, "29": 128, "15": 128},
    )
    rows += pipeline.run_stage(
        "coding",
        pipeline.CODING,
        {"72": 16, "133": 8, "29": 16, "15": 16},
        extra=(
            "--strategy", "swe_bench_toolcall",
            "--tools", "bash",
            "--agent-environment", "docker",
            "--max-steps", "250",
        ),
        config=pipeline.ROOT / "configs/evalscope_agent/swebench_verified_mini_docker.toml",
        agent_config=None,
    )
    rows += pipeline.run_stage(
        "terminal",
        pipeline.TERMINAL,
        {"72": 8, "133": 4, "29": 8, "15": 8},
        extra=("--no-agent-config",),
        config=pipeline.ROOT / "configs/evalscope_agent/terminal_bench_v2_1_docker.toml",
        agent_config=None,
    )
    tau_stages = (
        ("tau2", pipeline.TAU2, "git+https://github.com/sierra-research/tau2-bench@v0.2.0"),
        ("tau", pipeline.TAU, "git+https://github.com/sierra-research/tau-bench"),
        (
            "tau3",
            pipeline.TAU3,
            "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0",
        ),
    )
    for name, datasets, requirement in tau_stages:
        try:
            pipeline.prepare_tau(requirement, name)
        except Exception as error:  # noqa: BLE001 - retain the failure and continue
            print(f"ENVIRONMENT_PAUSED {name} error={error}", flush=True)
            rows.append({"stage": name, "environment_error": str(error)})
            continue
        rows += pipeline.run_stage(
            name,
            datasets,
            {"72": 16, "133": 8, "29": 16, "15": 16},
        )
    (pipeline.OUTROOT / "fc14-pipeline-summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
