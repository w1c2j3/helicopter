from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from helicopter_cli.evalscope_agent import (
    DEFAULT_CATALOG,
    build_evalscope_plan,
    format_agent_catalog,
    load_agent_catalog,
    _latest_evalscope_work_dir,
    _infer_plan,
    run_evalscope,
)
from helicopter_cli.__main__ import build_parser
from helicopter_cli.commands import build_infer_plan


ROOT = Path(__file__).resolve().parents[1]


def evalscope_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "model": "demo",
        "datasets": ["swe_bench_verified_agentic", "gaia"],
        "base_url": None,
        "api_key": None,
        "served_model_name": None,
        "eval_type": None,
        "mode": None,
        "framework": None,
        "agent_config": None,
        "strategy": None,
        "tools": None,
        "agent_environment": None,
        "agent_timeout": None,
        "max_steps": None,
        "limit": 2,
        "eval_batch_size": None,
        "generation_config": None,
        "dataset_args": None,
        "dataset_hub": None,
        "work_dir": None,
        "no_timestamp": None,
        "use_cache": None,
        "rerun_review": None,
        "enable_progress_tracker": None,
        "collect_perf": None,
        "debug": None,
        "ignore_errors": None,
        "binary": None,
        "dry_run": False,
        "no_server": False,
        "enable_auto_tool_choice": None,
        "tool_call_parser": None,
        "naive_chat_proxy": None,
        "report_only": False,
        "list_datasets": False,
    }
    values.update(overrides)
    return Namespace(**values)


class EvalScopeAgentTests(unittest.TestCase):
    def test_external_mode_is_a_supported_cli_choice(self) -> None:
        args = build_parser().parse_args(["eval", "evalscope", "demo", "general_fc", "--mode", "external"])
        self.assertEqual(args.mode, "external")

    def test_managed_rwkv_server_emits_native_tool_parser_flags(self) -> None:
        args = Namespace(
            model="demo",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            host=None,
            port=None,
            served_model_name=None,
            tensor_parallel_size=None,
            gpu_memory_utilization=None,
            max_model_len=None,
            max_num_seqs=None,
            max_num_batched_tokens=None,
            enable_auto_tool_choice=True,
            tool_call_parser=None,
            vllm_env=None,
        )
        config = {
            "models": {"demo": {"path": "/tmp/rwkv-demo.pth", "served_model_name": "demo-served"}},
            "infer": {},
        }

        plan = build_infer_plan(args, root=ROOT, env={}, config=config)

        self.assertIn("--enable-auto-tool-choice", plan.command)
        parser_index = plan.command.index("--tool-call-parser")
        self.assertEqual(plan.command[parser_index + 1], "rwkv")

    def test_managed_rwkv_server_allows_explicit_parser_override(self) -> None:
        args = Namespace(
            model="demo",
            dry_run=True,
            wkv_mode=None,
            emb_device=None,
            host=None,
            port=None,
            served_model_name=None,
            tensor_parallel_size=None,
            gpu_memory_utilization=None,
            max_model_len=None,
            max_num_seqs=None,
            max_num_batched_tokens=None,
            enable_auto_tool_choice=True,
            tool_call_parser="custom",
            vllm_env=None,
        )
        config = {"models": {"demo": {"path": "/tmp/rwkv-demo.pth"}}, "infer": {}}

        plan = build_infer_plan(args, root=ROOT, env={}, config=config)

        parser_index = plan.command.index("--tool-call-parser")
        self.assertEqual(plan.command[parser_index + 1], "custom")

    def test_native_evalscope_plan_enables_auto_tool_choice(self) -> None:
        args = evalscope_args(dry_run=True)
        config = {
            "models": {"demo": {"path": "/tmp/rwkv-demo.pth", "served_model_name": "demo-served"}},
            "evalscope": {"mode": "native"},
            "infer": {},
        }

        plan = _infer_plan(
            args,
            root=ROOT,
            env={},
            config=config,
            base_url="http://127.0.0.1:19329/v1",
        )

        assert plan is not None
        self.assertIn("--enable-auto-tool-choice", plan.command)
        parser_index = plan.command.index("--tool-call-parser")
        self.assertEqual(plan.command[parser_index + 1], "rwkv")

    def test_native_evalscope_rejects_naive_proxy(self) -> None:
        args = evalscope_args(mode="native", naive_chat_proxy=True)
        config = {
            "models": {"demo": {"served_model_name": "demo-served"}},
            "evalscope": {"mode": "native", "naive_chat_proxy": False},
        }

        with self.assertRaisesRegex(SystemExit, "requires the OpenAI tools/tool_choice"):
            run_evalscope(args, root=ROOT, env={}, config=config)

    def test_catalog_matches_supported_agent_dataset_shape(self) -> None:
        rows = load_agent_catalog(ROOT, DEFAULT_CATALOG)

        self.assertGreaterEqual(len(rows), 30)
        self.assertEqual(rows[0]["name"], "automation_bench")
        self.assertIn("swe_bench_verified_agentic", {row["name"] for row in rows})
        self.assertIn("Agent", rows[0]["categories"])
        self.assertIn("dataset\tdisplay_name\tcategories", format_agent_catalog(rows))

    def test_build_plan_uses_openai_endpoint_and_native_agent_config(self) -> None:
        config = {
            "models": {"demo": {"served_model_name": "demo-served"}},
            "lighteval": {"base_url": "http://127.0.0.1:8000/v1"},
            "evalscope": {
                "output_dir": "results/evalscope-test",
                "agent_config": {
                    "strategy": "swe_bench_toolcall",
                    "environment": "docker",
                    "max_steps": 250,
                },
                "generation_config": {"max_tokens": 4096, "temperature": 0.0},
                "dataset_args": {"swe_bench_verified_agentic": {"extra_params": {"build_docker_images": False}}},
            },
        }

        plan = build_evalscope_plan(evalscope_args(), root=ROOT, env={}, config=config)

        self.assertEqual(plan.command[:2], ["evalscope", "eval"])
        self.assertIn("--model", plan.command)
        self.assertIn("demo-served", plan.command)
        self.assertIn("--datasets", plan.command)
        self.assertEqual(
            plan.command[plan.command.index("--datasets") + 1 : plan.command.index("--work-dir")],
            ["swe_bench_verified_agentic", "gaia"],
        )
        agent_config = json.loads(plan.command[plan.command.index("--agent-config") + 1])
        self.assertEqual(agent_config["strategy"], "swe_bench_toolcall")
        self.assertEqual(agent_config["max_steps"], 250)
        self.assertIn('"max_tokens":4096', plan.command[plan.command.index("--generation-config") + 1])

    def test_bridge_cli_overrides_framework_and_timeout(self) -> None:
        config = {
            "models": {"demo": {"served_model_name": "demo-served"}},
            "lighteval": {"base_url": "https://example.test/v1"},
            "evalscope": {"output_dir": "results/evalscope-test", "mode": "bridge"},
        }

        plan = build_evalscope_plan(
            evalscope_args(
                datasets=["swe_bench_pro"],
                mode="bridge",
                framework="codex",
                agent_environment="docker",
                agent_timeout=1800,
            ),
            root=ROOT,
            env={},
            config=config,
        )

        agent_config = json.loads(plan.command[plan.command.index("--agent-config") + 1])
        self.assertEqual(agent_config["mode"], "external")
        self.assertEqual(agent_config["framework"], "codex")
        self.assertEqual(agent_config["environment"], "docker")
        self.assertEqual(agent_config["timeout"], 1800)
        self.assertEqual(plan.command[plan.command.index("--api-url") + 1], "https://example.test/v1")

    def test_latest_work_dir_resolves_timestamped_evalscope_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "20260726_120000" / "predictions"
            newer = root / "20260726_120001" / "predictions"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            self.assertEqual(_latest_evalscope_work_dir(root), newer.parent)


if __name__ == "__main__":
    unittest.main()
