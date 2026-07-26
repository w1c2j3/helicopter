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
)


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
    }
    values.update(overrides)
    return Namespace(**values)


class EvalScopeAgentTests(unittest.TestCase):
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
