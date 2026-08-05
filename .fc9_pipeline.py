import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/home/rwkv/chase/helicopter-e0eeddc")
PYTHON = ROOT / ".venv/bin/python"
EVALSCOPE = ROOT / ".venv/bin/evalscope"
CONFIG = ROOT / "configs/models/g1h-dual-replica.toml"
OUTROOT = Path("/home/rwkv/chase/eval-results/evalscope-g1h-20260804-fc12")
AGENT_CONFIG = Path("/home/rwkv/chase/eval-results/.fc9_agent_config.json")
GENERATION_CONFIG = Path("/home/rwkv/chase/eval-results/.fc9_generation_config.json")
MODEL_ARGS = Path("/home/rwkv/chase/eval-results/.fc9_model_args.json")
EXTERNAL_API_BASE = "https://next-token.cc/v1"
SKILLSBENCH_DIR = Path("/home/rwkv/chase/rwkv-skills/data/skillsbench")
MCP_ATLAS_URL = "http://127.0.0.1:1984"
UV = Path("/home/rwkv/.local/bin/uv")

MODELS = [
    ("72", "g1h-7.2b", 29572, 128),
    ("133", "g1h-13.3b", 29533, 64),
    ("29", "g1h-2.9b", 19429, 128),
    ("15", "g1h-1.5b", 19415, 128),
]

CORE = [
    "bfcl_v3", "bfcl_v4", "general_fc", "k2_verifier", "kimi_verifier",
    "minimax_verifier", "toolathlon",
]
MCP = ["mcp_atlas"]
SKILLSBENCH = ["skillsbench"]
GENERAL = [
    "browsecomp", "claw_eval", "gaia", "gdpval", "officeqa", "researchrubrics", "wide_search",
]
CODING = [
    "deep_swe", "swe_bench_lite_agentic", "swe_bench_multilingual_agentic",
    "swe_bench_pro", "swe_bench_verified_agentic", "swe_bench_verified_mini_agentic",
]
TERMINAL = ["terminal_bench_v2", "terminal_bench_v2_1"]
TAU2 = ["tau2_bench"]
TAU = ["tau_bench"]
TAU3 = ["tau3_bench"]

ENV = os.environ.copy()
ENV["PATH"] = str(ROOT / ".venv/bin") + ":" + ENV.get("PATH", "")
ENV["PYTHONPATH"] = ":".join(
    [str(ROOT / "src/cli"), str(ROOT / "src/scoreboard-server"), ENV.get("PYTHONPATH", "")]
)
ENV["HELICOPTER_EVAL_API_KEY"] = ENV.get("HELICOPTER_EVAL_API_KEY", "rwkv-skills")
EXTERNAL_API_KEY = ENV.get("HELICOPTER_EXTERNAL_API_KEY", "")


ROUTER_ARGS = [
    "--parallel-candidate-router",
    "--candidate-chunk-tools", "2",
    "--candidate-batch-size", "16",
    "--candidate-context-chars", "6000",
    "--candidate-prompt-max-chars", "12288",
    "--candidate-max-tokens", "2048",
    "--aggregate-max-tokens", "2048",
    "--candidate-max-candidates", "12",
    "--long-doc-min-chars", "6000",
    "--long-doc-max-chars", "1000",
    "--long-doc-overlap-lines", "3",
    "--long-doc-max-evidence-chunks", "4",
    "--long-doc-max-evidence-chars", "6000",
]


def dataset_args_for(datasets):
    """Build only benchmark-specific runtime arguments.

    The model's tool-call payload is still judged by EvalScope/BFCL.  These
    values only point adapters at their required local service/data or at the
    explicitly supplied external user/judge model.
    """

    result = {}
    if "skillsbench" in datasets:
        result["skillsbench"] = {
            "extra_params": {
                "tasks_dir": str(SKILLSBENCH_DIR),
                "skill_mode": "no-skill",
            }
        }
    if "mcp_atlas" in datasets:
        result["mcp_atlas"] = {
            "extra_params": {
                "mcp_server_url": MCP_ATLAS_URL,
                "filter_enabled_servers": True,
            }
        }
    for dataset in ("tau2_bench", "tau_bench", "tau3_bench"):
        if dataset not in datasets:
            continue
        extra = {
            "user_model": "gpt-4o-mini",
            "api_key": EXTERNAL_API_KEY,
            "api_base": EXTERNAL_API_BASE,
            "generation_config": {"temperature": 0.01, "max_tokens": 2048},
        }
        if dataset == "tau3_bench":
            extra["retrieval_config"] = "bm25"
        result[dataset] = {"extra_params": extra}
    return result


def launch(stage, tag, model, port, batch, datasets, extra=(), config=None, agent_config=AGENT_CONFIG):
    work = OUTROOT / f"{stage}-fc12-parallel-longctx-g1h-{tag}b"
    work.mkdir(parents=True, exist_ok=True)
    args = [
        str(PYTHON), "-m", "helicopter_cli", "eval", "evalscope",
        "--config", str(config or CONFIG), model, *datasets,
        "--binary", str(EVALSCOPE),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--api-key", "rwkv-skills",
        "--mode", "native",
        "--no-server",
        "--no-naive-chat-proxy",
        "--parallel-candidate-router",
        "--generation-config", str(GENERATION_CONFIG),
        "--model-args", str(MODEL_ARGS),
        "--eval-batch-size", str(batch),
        "--work-dir", str(work),
        "--ignore-errors",
        "--scoreboard",
        "--judge-strategy", "auto",
        *ROUTER_ARGS[1:],
        *extra,
    ]
    dataset_args = dataset_args_for(datasets)
    if dataset_args:
        insert_at = args.index("--generation-config")
        args[insert_at:insert_at] = [
            "--dataset-args",
            json.dumps(dataset_args, ensure_ascii=False, separators=(",", ":")),
        ]
    if EXTERNAL_API_KEY:
        insert_at = args.index("--generation-config")
        args[insert_at:insert_at] = [
            "--judge-model-args",
            json.dumps(
                {
                    "model_id": "gpt-4o-mini",
                    "api_url": EXTERNAL_API_BASE,
                    "api_key": EXTERNAL_API_KEY,
                    "generation_config": {"temperature": 0.01, "max_tokens": 2048},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    if agent_config is not None:
        insert_at = args.index("--generation-config")
        args[insert_at:insert_at] = ["--agent-config", str(agent_config)]
    log_path = work / "run.log"
    log = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(args, cwd=str(ROOT), env=ENV, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, work, args


def run_stage(name, datasets, batches, extra=(), config=None, agent_config=AGENT_CONFIG):
    print(f"STAGE_START {name} datasets={','.join(datasets)}", flush=True)
    running = []
    for tag, model, port, _ in MODELS:
        proc, log, work, args = launch(name, tag, model, port, batches[tag], datasets, extra, config, agent_config)
        running.append((tag, proc, log, work, args))
        print(f"LAUNCHED {name} {tag} pid={proc.pid} work={work}", flush=True)

    rows = []
    for tag, proc, log, work, args in running:
        rc = proc.wait()
        log.close()
        rows.append({"stage": name, "model": tag, "returncode": rc, "work_dir": str(work)})
        print(f"DONE {name} {tag} returncode={rc} work={work}", flush=True)
    return rows


def prepare_tau(requirement, label):
    """Install one tau family into the shared EvalScope venv before its stage."""

    if not UV.is_file():
        raise RuntimeError(f"uv not found: {UV}")
    print(f"TAU_ENV_START {label} requirement={requirement}", flush=True)
    subprocess.run(
        [str(UV), "pip", "uninstall", "--python", str(PYTHON), "-y", "tau2"],
        cwd=str(ROOT),
        env=ENV,
        check=False,
    )
    completed = subprocess.run(
        [str(UV), "pip", "install", "--python", str(PYTHON), requirement],
        cwd=str(ROOT),
        env=ENV,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"tau environment setup failed for {label}: returncode={completed.returncode}")
    print(f"TAU_ENV_READY {label}", flush=True)


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += run_stage("core", CORE, {"72": 128, "133": 64, "29": 128, "15": 128})
    rows += run_stage("mcp", MCP, {"72": 64, "133": 32, "29": 64, "15": 64})
    rows += run_stage("skillsbench", SKILLSBENCH, {"72": 8, "133": 4, "29": 8, "15": 8})
    rows += run_stage("general", GENERAL, {"72": 128, "133": 64, "29": 128, "15": 128})
    rows += run_stage(
        "coding",
        CODING,
        {"72": 32, "133": 16, "29": 16, "15": 16},
        extra=(
            "--strategy", "swe_bench_toolcall",
            "--tools", "bash",
            "--agent-environment", "docker",
            "--max-steps", "250",
        ),
        config=ROOT / "configs/evalscope_agent/swebench_verified_mini_docker.toml",
        agent_config=None,
    )
    rows += run_stage(
        "terminal",
        TERMINAL,
        {"72": 8, "133": 4, "29": 8, "15": 8},
        extra=("--no-agent-config",),
        config=ROOT / "configs/evalscope_agent/terminal_bench_v2_1_docker.toml",
        agent_config=None,
    )
    tau_stages = (
        ("tau2", TAU2, "git+https://github.com/sierra-research/tau2-bench@v0.2.0"),
        ("tau", TAU, "git+https://github.com/sierra-research/tau-bench"),
        ("tau3", TAU3, "tau2[knowledge] @ git+https://github.com/sierra-research/tau2-bench@v1.0.0"),
    )
    for name, datasets, requirement in tau_stages:
        try:
            prepare_tau(requirement, name)
        except Exception as error:  # noqa: BLE001 - retain the environment failure and continue
            print(f"ENVIRONMENT_PAUSED {name} error={error}", flush=True)
            rows.append({"stage": name, "environment_error": str(error)})
            continue
        rows += run_stage(
            name,
            datasets,
            {"72": 32, "133": 16, "29": 16, "15": 16},
        )
    (OUTROOT / "fc12-pipeline-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
