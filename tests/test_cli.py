from __future__ import annotations

import base64
from collections import Counter
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import bigbench as bigbench_tasks
from lighteval.tasks.tasks.coqa import coqa_first_question as coqa_task
from lighteval.tasks.tasks.glue import rte_prompt as super_glue_rte_prompt
from lighteval.tasks.tasks.ifbench import instructions as ifbench_instructions
from lighteval.tasks.tasks.math_500 import math_500 as math_500_task, math_500_prompt
from lighteval.tasks.tasks.mmlu_pro import mmlu_pro as mmlu_pro_task, mmlu_pro_prompt_function
from lighteval.tasks.tasks.piqa import piqa as piqa_task
from lighteval.tasks.tasks.pubmedqa import pubmed_qa_prompt, pubmedqa as pubmedqa_task
from lighteval.tasks.tasks.simpleqa import simpleqa_prompt
from lighteval.tasks.tasks.truthfulqa import truthful_qa_generative_prompt

from helicopter_cli import __main__ as helicopter_main
from helicopter_cli import (
    agent_format,
    agent_harness,
    benchmark_specs,
    commands,
    config,
    env,
    eval_batch,
    eval_run,
    function_calling,
    rwkv_config,
    lighteval_g1h_policy,
    lighteval_answer_adapters,
    lighteval_db_pipeline,
    lighteval_export,
    lighteval_dataset_resilience,
    lighteval_raw_completion,
    lighteval_rwkv_skills_tasks,
    lighteval_tasks,
    performance,
    scoreboard_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "configs/example.toml"


class Famous120BenchmarkConfigTests(unittest.TestCase):
    @unittest.skip("legacy famous120 aggregate was removed")
    def test_famous120_is_one_valid_toml_per_benchmark(self) -> None:
        specs = benchmark_specs.load_benchmark_index(
            ROOT / "configs/benchmarks/famous120.toml"
        )
        self.assertEqual(len(specs), 120)
        self.assertEqual(
            Counter(spec["benchmark"]["field"] for spec in specs),
            Counter({
                "knowledge": 30,
                "math": 30,
                "coding": 30,
                "instruction_following": 30,
            }),
        )
        self.assertEqual(len({spec["_path"] for spec in specs}), 120)
        self.assertTrue(
            all(
                spec["scoring"]
                == {"provider": "lighteval", "metric_source": "task_default"}
                for spec in specs
            )
        )
        self.assertTrue(all(spec["evaluation"]["avg_k"] == 8 for spec in specs))

    def test_each_standalone_benchmark_config_loads_independently(self) -> None:
        paths = sorted((ROOT / "configs/benchmarks/g1h").glob("*/*.toml"))
        self.assertGreaterEqual(len(paths), 400)
        for path in paths:
            with self.subTest(path=path.name):
                loaded, _ = config.load_config(ROOT, str(path))
                self.assertEqual(len(loaded["_benchmark_specs"]), 1)
                self.assertNotIn("g1h", str(loaded.get("lighteval", {})))
                self.assertNotIn("models", loaded)

    def test_model_catalog_is_selected_outside_benchmark_config(self) -> None:
        loaded, _ = config.load_config(
            ROOT,
            "configs/benchmarks/g1h/knowledge/046_gpqa_diamond.toml",
        )
        config.merge_model_catalog(
            loaded,
            root=ROOT,
            catalog_path="configs/models/g1h-dual-replica.toml",
        )
        self.assertEqual(
            set(loaded["models"]),
            {"deployed", "g1h-1.5b", "g1h-2.9b", "g1h-7.2b", "g1h-13.3b"},
        )

    def test_gpqa_token_budget_is_resolved_per_prompt_mode(self) -> None:
        config_path = "configs/benchmarks/g1h/knowledge/046_gpqa_diamond.toml"
        loaded, _ = config.load_config(ROOT, config_path)
        config.merge_model_catalog(
            loaded,
            root=ROOT,
            catalog_path="configs/models/g1h-dual-replica.toml",
        )
        plan = commands.build_lighteval_plan(
            lighteval_args(model="deployed", tasks="gpqa:diamond|0", config=config_path),
            root=ROOT,
            env={},
            config=loaded,
        )
        tasks = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]
        self.assertEqual(tasks["gpqa:diamond"]["sampling"]["max_tokens"], 1024)

    def test_gpqa_uses_domain_budget_for_each_prompt_mode(self) -> None:
        config_path = "configs/benchmarks/g1h/knowledge/046_gpqa_diamond.toml"
        loaded, _ = config.load_config(ROOT, config_path)
        config.merge_model_catalog(
            loaded,
            root=ROOT,
            catalog_path="configs/models/g1h-dual-replica.toml",
        )
        plan = commands.build_lighteval_plan(
            lighteval_args(model="deployed", tasks="gpqa:diamond|0", config=config_path),
            root=ROOT,
            env={},
            config=loaded,
        )
        tasks = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]
        self.assertEqual(tasks["gpqa:diamond"]["sampling"]["context_budget"], 10240)

class NativeLightEvalTaskCompatibilityTests(unittest.TestCase):
    def test_math_adapter_canonicalizes_prediction_and_gold_forms(self) -> None:
        for value in (r"$\boxed{36}$", r"\boxed{36}", "36", r"\(36\)"):
            with self.subTest(value=value):
                self.assertEqual(
                    lighteval_answer_adapters.extract_math_answer(value),
                    "$36$",
                )
        self.assertEqual(
            lighteval_answer_adapters.adapt_answer(
                r"Reasoning\n\boxed{36}",
                domain="math",
                request_format="math_boxed",
            ),
            "$36$",
        )

    def test_math_prompt_gold_uses_the_same_type_level_adapter(self) -> None:
        prompt = lighteval_g1h_policy._wrap_prompt(
            lambda line, task_name=None: Doc(
                task_name=task_name,
                query=line["question"],
                choices=[r"$\boxed{36}$"],
                gold_index=0,
            ),
            canonical_name="gsm8k",
            policy={},
        )
        with mock.patch.dict(
            os.environ,
            {
                "HELICOPTER_PROMPT_TEMPLATE": "raw",
                "HELICOPTER_LIGHTEEVAL_TASK_REQUEST_POLICY": json.dumps(
                    {"tasks": {"gsm8k": {"format": "math_boxed"}}}
                ),
            },
        ):
            doc = prompt({"question": "q"}, task_name="gsm8k")
        self.assertEqual(doc.choices, ["$36$"])

    def test_all_reference_forms_are_normalized_before_native_scoring(self) -> None:
        math_doc = Doc(query="q", choices=[r"$\boxed{4000}$"], gold_index=0)
        lighteval_g1h_policy._normalize_doc_references(
            math_doc,
            domain="math",
            request_format="math_boxed",
        )
        self.assertEqual(math_doc.choices, ["$4000$"])

        judge_doc = Doc(
            query="q",
            choices=["Judgement: Yes"],
            gold_index=0,
            specific={
                "references": ["\u003c think \u003eold\u003c/think\u003eJudgement: Yes"],
                "reference": "\u003c think \u003eold\u003c/think\u003eJudgement: Yes",
            },
        )
        lighteval_g1h_policy._normalize_doc_references(
            judge_doc,
            domain="instruction_following",
            request_format="choice",
        )
        self.assertEqual(judge_doc.choices, ["Judgement: Yes"])
        self.assertEqual(judge_doc.specific["references"], ["Judgement: Yes"])
        self.assertEqual(judge_doc.specific["reference"], "Judgement: Yes")

    def test_math500_restores_official_prompt_and_solution_gold(self) -> None:
        doc = math_500_prompt(
            {
                "problem": "  Solve $x+1=2$.\r\n",
                "answer": "1",
                "solution": "A worked solution ending in a distractor 2.",
            },
            task_name="math_500",
        )

        self.assertEqual(doc.query, "Solve $x+1=2$.")
        self.assertEqual(doc.choices, [r"$\boxed{1}$"])
        self.assertEqual(doc.gold_index, 0)
        self.assertEqual(math_500_task.version, 3)

    def test_mmlu_pro_native_task_keeps_only_raw_question_and_choices(self) -> None:
        doc = mmlu_pro_prompt_function(
            {
                "question": "Which option is correct?",
                "options": ["First", "Second", "Third"],
                "answer_index": 1,
            },
            task_name="mmlu_pro",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(
            doc.query,
            "Which option is correct?\nA. First\nB. Second\nC. Third",
        )
        self.assertEqual(doc.choices, ["A", "B", "C"])
        self.assertEqual(doc.gold_index, 1)
        self.assertIsNone(doc.instruction)
        self.assertNotIn("Think step by step", doc.query)
        self.assertNotIn("Answer:", doc.query)
        self.assertEqual(mmlu_pro_task.version, 1)

    def test_truthfulqa_generation_prompt_does_not_require_mc_fields(self) -> None:
        doc = truthful_qa_generative_prompt(
            {
                "question": "What is true?",
                "correct_answers": ["The supported answer"],
                "incorrect_answers": ["The unsupported answer"],
            },
            task_name="truthfulqa:gen",
        )

        self.assertEqual(doc.query, "What is true?")
        self.assertEqual(doc.choices, ["The supported answer.", "I have no comment.", "The unsupported answer."])
        self.assertEqual(doc.gold_index, [0, 1])

    def test_selected_bigbench_tasks_use_available_validation_split(self) -> None:
        for task in (
            bigbench_tasks.snarks,
            bigbench_tasks.tellmewhy,
            bigbench_tasks.tense,
            bigbench_tasks.timedial,
            bigbench_tasks.winowhy,
        ):
            with self.subTest(task=task.name):
                self.assertEqual(task.hf_avail_splits, ("train", "validation"))
                self.assertEqual(task.evaluation_splits, ("validation",))

    def test_piqa_uses_script_free_official_lighteval_mirror(self) -> None:
        self.assertEqual(piqa_task.hf_repo, "lighteval/piqa")
        self.assertEqual(piqa_task.evaluation_splits, ("validation",))

    def test_coqa_formatter_lists_are_flattened_with_stable_ids(self) -> None:
        task = LightevalTask(coqa_task)
        task.dataset = {
            "validation": [
                {
                    "story": "A short story.",
                    "questions": ["First?", "Second?"],
                    "answers": {"input_text": ["one", "two"]},
                }
            ]
        }

        docs = task._get_docs_from_split(["validation"])

        self.assertEqual([doc.id for doc in docs], ["0:0", "0:1"])
        self.assertEqual([doc.choices for doc in docs], [["one"], ["two"]])
        self.assertTrue(docs[0].query.endswith("First?"))
        self.assertTrue(docs[1].query.endswith("Second?"))
        self.assertEqual(docs[0].stop_sequences, coqa_task.stop_sequence)

    def test_simpleqa_prompt_uses_official_free_response_fields(self) -> None:
        doc = simpleqa_prompt(
            {"problem": "Who received the award?", "answer": "The recipient"},
            task_name="simpleqa",
        )

        self.assertEqual(doc.query, "Who received the award?")
        self.assertEqual(doc.choices, ["The recipient"])
        self.assertEqual(doc.gold_index, 0)

    def test_super_glue_rte_uses_premise_hypothesis_schema(self) -> None:
        doc = super_glue_rte_prompt(
            {
                "premise": "Bacteria mutate faster than new antibiotics are developed.",
                "hypothesis": "Bacteria is winning the war against antibiotics.",
                "label": 0,
            },
            task_name="super_glue:rte",
        )

        self.assertEqual(
            doc.query,
            "Bacteria mutate faster than new antibiotics are developed.\n"
            "Bacteria is winning the war against antibiotics.",
        )
        self.assertEqual(doc.choices, [" True", " False"])
        self.assertEqual(doc.gold_index, 0)

    def test_pubmedqa_uses_namespaced_source_and_current_fields(self) -> None:
        self.assertEqual(pubmedqa_task.hf_repo, "qiaojin/PubMedQA")
        doc = pubmed_qa_prompt(
            {
                "question": "Is the claim supported?",
                "context": {"contexts": ["First abstract sentence.", "Second sentence."]},
                "final_decision": "yes",
            },
            task_name="pubmedqa",
        )

        self.assertEqual(doc.query, "Is the claim supported?\nFirst abstract sentence.\nSecond sentence.")
        self.assertEqual(doc.choices, ["yes"])
        self.assertEqual(doc.gold_index, 0)


class G1hConfigTests(unittest.TestCase):
    POLICY = {
        "prompt_style": "naive",
        "zero_shot": True,
        "avg_k": 8,
        "rollout_n": 8,
        "generation_size": 4096,
        "gpass_k": 16,
        "gpass_n": 48,
        "gpass_generation_size": 8192,
        "long_rollout_tasks": [],
        "no_cot_tasks": [],
    }
    def test_avg_k_must_be_declared_in_toml_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "avg_k must be set"):
            rwkv_config.normalize_policy({"metric": "avg", "prompt_style": "naive"})


    def test_variant_selection_prefers_avg_then_gpass(self) -> None:
        selected = rwkv_config.select_task_specs(
            ["aime24", "aime24_avg", "aime24_gpassk", "math_500"],
            self.POLICY,
        )
        self.assertEqual(selected, [("aime24_avg", "0"), ("math_500", "0")])

    def test_variant_selection_falls_back_to_gpass(self) -> None:
        selected = rwkv_config.select_task_specs(["toy", "toy_gpassk"], self.POLICY)
        self.assertEqual(selected, [("toy_gpassk", "0")])

    def test_prompt_profiles_are_configurable(self) -> None:
        naive = rwkv_config.format_query("Q", canonical_name="math_500", policy=self.POLICY)
        normal_policy = {**self.POLICY, "prompt_style": "normal"}
        normal = rwkv_config.format_query("Q", canonical_name="math_500", policy=normal_policy)
        self.assertEqual(naive, "User: Q\n\nAssistant: <think")
        self.assertEqual(normal, "User✿Q✿\nBot✿<think")


    def test_alias_specs_make_zero_shot_explicit(self) -> None:
        specs = rwkv_config.alias_task_specs([("math_500", "5")], self.POLICY)
        self.assertEqual(specs, ["g1h__math_500|0"])


class BenchmarkSamplingPolicyTests(unittest.TestCase):
    def test_small_benchmark_is_complete(self) -> None:
        self.assertIsNone(
            lighteval_db_pipeline.auto_sample_count(
                document_count=541,
                rollout_n=8,
                target_generations=5000,
                large_generation_threshold=20000,
                large_sample_rate=0.2,
            )
        )

    def test_medium_benchmark_targets_five_thousand(self) -> None:
        self.assertEqual(
            lighteval_db_pipeline.auto_sample_count(
                document_count=1319,
                rollout_n=8,
                target_generations=5000,
                large_generation_threshold=20000,
                large_sample_rate=0.2,
            ),
            625,
        )

    def test_large_benchmark_uses_configured_fraction(self) -> None:
        self.assertEqual(
            lighteval_db_pipeline.auto_sample_count(
                document_count=5000,
                rollout_n=8,
                target_generations=5000,
                large_generation_threshold=20000,
                large_sample_rate=0.2,
            ),
            1000,
        )

    def test_prefilled_reasoning_is_removed_for_scoring(self) -> None:
        self.assertEqual(
            lighteval_db_pipeline.strip_prefilled_reasoning(
                "reasoning steps</think>Final answer"
            ),
            "Final answer",
        )
        self.assertEqual(
            lighteval_db_pipeline.strip_prefilled_reasoning(
                "<think>reasoning steps</think>Final answer"
            ),
            "<think>reasoning steps</think>Final answer",
        )
        self.assertEqual(
            lighteval_db_pipeline.strip_prefilled_reasoning(
                "quoted instruction</think>actual reasoning</think>Final answer",
                force=True,
            ),
            "Final answer",
        )
        self.assertEqual(
            lighteval_db_pipeline.strip_prefilled_reasoning("unfinished reasoning"),
            "unfinished reasoning",
        )
        self.assertTrue(
            lighteval_db_pipeline.has_unclosed_reasoning_prefill("Assistant: <think")
        )
        self.assertFalse(
            lighteval_db_pipeline.has_unclosed_reasoning_prefill(
                "Assistant: <think></think>"
            )
        )
        self.assertTrue(
            lighteval_db_pipeline.has_empty_reasoning_prefill(
                "Assistant: <think></think"
            )
        )
        self.assertTrue(
            lighteval_db_pipeline.has_empty_reasoning_prefill(
                "Bot✿<think></think"
            )
        )
        self.assertFalse(
            lighteval_db_pipeline.has_empty_reasoning_prefill(
                "Assistant: <think></think>"
            )
        )

    def test_pipeline_scores_only_answer_after_prefilled_reasoning(self) -> None:
        pipeline = SimpleNamespace(
            pipeline_parameters=SimpleNamespace(
                remove_reasoning_tags=True,
                reasoning_tags=[("<think>", "</think>")],
            )
        )
        response = ModelResponse(
            text=[
                "reasoning</think>Answer",
                "<think>reasoning</think>Answer 2",
                "reasoning that quotes <think>example</think>Final answer",
                "unfinished",
            ],
            input="Assistant: <think",
        )

        lighteval_db_pipeline._post_process_outputs(
            pipeline,
            {"GENERATIVE": [response]},
        )

        self.assertEqual(
            response.text_post_processed,
            ["Answer", "Answer 2", "Final answer", "unfinished"],
        )
        self.assertEqual(
            response.text,
            [
                "reasoning</think>Answer",
                "<think>reasoning</think>Answer 2",
                "reasoning that quotes <think>example</think>Final answer",
                "unfinished",
            ],
        )
        self.assertEqual(
            scoreboard_bridge._answer({"text_post_processed": "", "text": "raw"}), ""
        )

        closed_prompt_response = ModelResponse(
            text=["def solve(): return 1"],
            input="Assistant: <think></think>\n```python",
        )
        lighteval_db_pipeline._post_process_outputs(
            pipeline,
            {"GENERATIVE": [closed_prompt_response]},
        )
        self.assertEqual(
            closed_prompt_response.text_post_processed,
            ["def solve(): return 1"],
        )

        # NoCoT owns the empty think block in the prompt. Its deliberately
        # incomplete closing tag is completed and removed by the request
        # adapter, so the entire stored continuation is the answer even when
        # LightEval's native reasoning remover produces an empty string.
        for prompt in (
            "Assistant: <think></think",
            "Bot✿<think></think",
        ):
            nocot_response = ModelResponse(text=["ANSWER: 52"], input=prompt)
            with mock.patch.object(
                lighteval_db_pipeline,
                "_ORIGINAL_POST_PROCESS_OUTPUTS",
                side_effect=lambda _pipeline, responses: setattr(
                    responses["GENERATIVE"][0], "text_post_processed", [""]
                ),
            ):
                lighteval_db_pipeline._post_process_outputs(
                    pipeline,
                    {"GENERATIVE": [nocot_response]},
                )
            self.assertEqual(nocot_response.text_post_processed, ["ANSWER: 52"])

    def test_code_fence_normalization_preserves_official_last_block(self) -> None:
        self.assertEqual(
            lighteval_db_pipeline.normalize_code_fences("plain text"),
            "plain text",
        )
        single = "explanation\n```python\nprint(1)"
        self.assertEqual(
            lighteval_db_pipeline.normalize_code_fences(single),
            single + "\n```",
        )
        complete = "```python\nprint(1)\n```"
        self.assertEqual(
            lighteval_db_pipeline.normalize_code_fences(complete),
            complete,
        )
        multiple = "```text\nexample\n```\n```python\nprint(2)\n```"
        self.assertEqual(
            lighteval_db_pipeline.normalize_code_fences(multiple),
            "```python\nprint(2)\n```",
        )

    def test_code_fence_normalization_is_selected_by_toml_format(self) -> None:
        pipeline = SimpleNamespace(
            pipeline_parameters=SimpleNamespace(remove_reasoning_tags=False)
        )
        response = ModelResponse(
            text=["explanation\n```python\nprint(1)"],
            input="Assistant: <think></think",
        )
        policy = json.dumps({"tasks": {"lcb:codegeneration_v5": {"format": "code"}}})
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": policy},
            clear=False,
        ):
            lighteval_db_pipeline._post_process_outputs(
                pipeline,
                {"GENERATIVE": [response]},
            )
        self.assertEqual(
            response.text_post_processed,
            ["```\nprint(1)\n```"],
        )

    def test_invalid_large_sample_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "large_benchmark_sample_rate"):
            rwkv_config.normalize_policy(
                {
                    **G1hConfigTests.POLICY,
                    "large_benchmark_sample_rate": 1.5,
                }
            )


def load_example_config() -> dict[str, object]:
    loaded, _ = config.load_config(ROOT, str(EXAMPLE_CONFIG))
    return loaded


def infer_args(**overrides: object) -> Namespace:
    values = {
        "model": "g1g-1.5b",
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "host": None,
        "port": None,
        "served_model_name": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_model_len": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "enable_auto_tool_choice": None,
        "vllm_env": None,
    }
    values.update(overrides)
    return Namespace(**values)


def takeoff_args(**overrides: object) -> Namespace:
    values = {
        "algorithm": "grpo",
        "model": "g1g-1.5b",
        "dataset": "gsm8k",
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "num_nodes": None,
        "num_devices": None,
        "override": None,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_args(**overrides: object) -> Namespace:
    values = {
        "backend": "endpoint-litellm",
        "model": "g1g-1.5b",
        "tasks": "gsm8k",
        "model_args": None,
        "lighteval_model_name": None,
        "base_url": None,
        "provider": None,
        "api_key": None,
        "prompt_mode": None,
        "concurrent_requests": None,
        "max_model_length": None,
        "max_new_tokens": None,
        "max_samples": None,
        "output_dir": None,
        "dataset_loading_processes": None,
        "num_fewshot_seeds": None,
        "custom_tasks": None,
        "load_tasks_multilingual": None,
        "save_details": None,
        "push_to_hub": None,
        "public_run": None,
        "results_org": None,
        "job_id": None,
        "extra": None,
        "performance_output": None,
        "metrics_url": None,
        "scoreboard_task_id": None,
    }
    values.update(overrides)
    return Namespace(**values)


def perf_args(**overrides: object) -> Namespace:
    values = {
        "model": "g1d-0.4b",
        "dry_run": True,
        "base_url": None,
        "api_key": None,
        "served_model_name": None,
        "profile": "decode",
        "prompt_tokens": None,
        "output_tokens": None,
        "requests": None,
        "concurrency": None,
        "request_rate": None,
        "timeout": None,
        "ignore_eos": None,
        "output": None,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_tasks_args(**overrides: object) -> Namespace:
    values = {
        "task_action": "list",
        "tasks": None,
        "custom_tasks": None,
        "load_tasks_multilingual": None,
        "num_samples": None,
        "show_config": None,
        "output": None,
        "format": "text",
        "contains": None,
        "limit": None,
        "include_supersets": None,
        "source": None,
        "source_format": "auto",
        "candidate_limit": 5,
    }
    values.update(overrides)
    return Namespace(**values)


def lighteval_export_args(**overrides: object) -> Namespace:
    values = {
        "details": ["results/lighteval/details/run"],
        "output": None,
        "format": "jsonl",
    }
    values.update(overrides)
    return Namespace(**values)


def command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[item] = command[index + 1]
            index += 2
        else:
            options[item] = True
            index += 1
    return options


def hydra_pairs(plan: commands.CommandPlan) -> list[tuple[str, str]]:
    pairs = []
    for item in plan.command[3:]:
        if "=" in item:
            key, value = item.split("=", 1)
            pairs.append((key, value))
    return pairs


def hydra_map(plan: commands.CommandPlan) -> dict[str, str]:
    return dict(hydra_pairs(plan))


def hydra_values(plan: commands.CommandPlan, key: str) -> list[str]:
    return [value for pair_key, value in hydra_pairs(plan) if pair_key == key]


def build_takeoff_plan(
    loaded_config: dict[str, object],
    *,
    args: Namespace | None = None,
    loaded_env: dict[str, str] | None = None,
    venv_python: Path | None = None,
) -> commands.CommandPlan:
    if loaded_env is None:
        loaded_env = {"WEIGHT_PATH": "/weights/RWKV", "DATASETS_PATH": "/datasets"}
    if args is None:
        args = takeoff_args()
    if venv_python is None:
        venv_python = ROOT / ".venv/bin/python"
    original_exists = Path.exists
    with mock.patch.object(Path, "exists", autospec=True) as exists:
        exists.side_effect = lambda path: True if path == venv_python else original_exists(path)
        return commands.build_takeoff_plan(args, root=ROOT, env=loaded_env, config=loaded_config)


class RawCompletionTests(unittest.TestCase):

    def test_lighteval_instruction_is_not_injected_into_raw_query(self) -> None:
        instruction = "You are answering a multiple choice question.\n"
        with_instruction = Doc(
            query="Question: Raw question\nAnswer:",
            choices=[" A", " B"],
            gold_index=0,
            instruction=instruction,
        )
        self.assertEqual(
            lighteval_raw_completion._benchmark_query(with_instruction),
            "Question: Raw question\nAnswer:",
        )

        duplicated = Doc(
            query=instruction + "Question: Raw question\nAnswer:",
            choices=[" A", " B"],
            gold_index=0,
            instruction=instruction,
        )
        self.assertEqual(
            lighteval_raw_completion._benchmark_query(duplicated),
            instruction + "Question: Raw question\nAnswer:",
        )

    def test_raw_question_guard_rejects_added_task_cues(self) -> None:
        policy = {
            "tasks": {
                "mmlu_pro": {
                    "benchmark_config_path": "configs/benchmarks/famous120/knowledge/01_mmlu_pro.toml"
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "added cue"):
                lighteval_raw_completion._validate_raw_question_contract(
                    "g1h__mmlu_pro|0",
                    "User: Answer the following multiple choice question.\nAssistant: <think",
                )
            with self.assertRaisesRegex(RuntimeError, "trailing Answer"):
                lighteval_raw_completion._validate_raw_question_contract(
                    "g1h__mmlu_pro|0",
                    "User: Raw question\nA. one\nB. two\nAnswer:\n\nAssistant: <think",
                )
            lighteval_raw_completion._validate_raw_question_contract(
                "g1h__mmlu_pro|0",
                "User: Raw question\nA. one\nB. two\nAssistant: <think",
            )

    def test_structured_code_prefill_is_restored_for_official_extractors(self) -> None:
        self.assertEqual(
            lighteval_raw_completion._restore_structured_prefill(
                "print(1)\n```", "User: Q\nAssistant: <think>\n</think>\n```python"
            ),
            "```python\nprint(1)\n```",
        )
        self.assertEqual(
            lighteval_raw_completion._restore_structured_prefill(
                "diff --git a/x b/x\n```", "User: Q\nAssistant: <think>\n</think>\n```diff"
            ),
            "```diff\ndiff --git a/x b/x\n```",
        )

        self.assertEqual(
            lighteval_raw_completion._restore_structured_prefill(
                "print(2)",
                "User: Q\nAssistant: <think>\n</think>\n```python",
                restore_stop_fence=True,
            ),
            "```python\nprint(2)\n```",
        )
        self.assertEqual(
            lighteval_raw_completion._restore_structured_prefill(
                "fn main() {}",
                "User: Q\nAssistant: <think>\n</think>\n```",
                restore_stop_fence=True,
            ),
            "```\nfn main() {}\n```",
        )

    def test_avg_logprob_mcq_choices_are_serialized_before_generation(self) -> None:
        from lighteval.tasks.tasks.arc import arc_challenge

        configured = lighteval_g1h_policy._policy_config(
            arc_challenge,
            canonical_name="arc:challenge",
            policy=G1hConfigTests.POLICY,
        )
        with mock.patch.dict(os.environ, {"HELICOPTER_PROMPT_TEMPLATE": "User: {query}"}, clear=False):
            converted = configured.prompt_function(
                {
                    "question": "Which gas do plants absorb?",
                    "choices": {
                        "text": ["carbon dioxide", "oxygen", "nitrogen", "helium"],
                        "label": ["A", "B", "C", "D"],
                    },
                    "answerKey": "A",
                },
                "arc:challenge",
            )

        self.assertEqual(
            converted.query,
            "Question: Which gas do plants absorb?\nAnswer:",
        )
        self.assertEqual(
            converted.choices,
            [" carbon dioxide", " oxygen", " nitrogen", " helium"],
        )
        self.assertNotIn("helicopter_generated_mcq", converted.specific or {})
        self.assertEqual(configured.metrics[0].category, lighteval_g1h_policy.SamplingMethod.GENERATIVE)
        self.assertIsInstance(configured.metrics[0].sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(configured.metrics[0].sample_level_fn.n, 8)

    def test_ifbench_resource_check_never_downloads_during_scoring(self) -> None:
        with mock.patch.object(ifbench_instructions.nltk.data, "find"), mock.patch.object(
            ifbench_instructions.spacy,
            "load",
        ), mock.patch.object(ifbench_instructions.nltk, "download") as nltk_download:
            ifbench_instructions._ensure_local_resources()

        nltk_download.assert_not_called()

    def test_ifbench_missing_resource_fails_before_scoring_without_network(self) -> None:
        with mock.patch.object(
            ifbench_instructions.nltk.data,
            "find",
            side_effect=LookupError,
        ), mock.patch.object(ifbench_instructions.nltk, "download") as nltk_download, self.assertRaisesRegex(
            RuntimeError,
            "requires preinstalled NLTK resources",
        ):
            ifbench_instructions._ensure_local_resources()

        nltk_download.assert_not_called()

    def test_ifbench_emoji_checker_rejects_punctuation_only_sentence_without_crashing(self) -> None:
        checker = object.__new__(ifbench_instructions.EmojiSentenceChecker)

        self.assertFalse(checker.check_following('Hello. "!"'))
        self.assertFalse(checker.check_following('"!"'))

    def test_ifbench_paragraph_checker_ignores_punctuation_only_paragraphs(self) -> None:
        checker = object.__new__(ifbench_instructions.ParagraphLastFirstWordMatchChecker)

        self.assertTrue(checker.check_following("Alpha body Alpha\n---"))
        self.assertTrue(checker.check_following("***\nBeta body Beta"))
        self.assertFalse(checker.check_following("Alpha body Omega\n---"))

    def test_ifbench_alphabet_loop_checker_rejects_punctuation_only_response(self) -> None:
        checker = object.__new__(ifbench_instructions.AlphabetLoopChecker)

        self.assertFalse(checker.check_following("---"))
        self.assertFalse(checker.check_following("***"))

    def test_ifbench_sentence_chain_checker_rejects_punctuation_only_sentence(self) -> None:
        checker = object.__new__(ifbench_instructions.LastWordFirstNextChecker)

        with mock.patch.object(
            ifbench_instructions.instructions_util,
            "split_into_sentences",
            return_value=["Alpha beta.", "beta gamma."],
        ):
            self.assertTrue(checker.check_following("ignored"))
        with mock.patch.object(
            ifbench_instructions.instructions_util,
            "split_into_sentences",
            return_value=["Alpha beta.", "?", "beta gamma."],
        ):
            self.assertFalse(checker.check_following("ignored"))

    def test_ifeval_stopwords_are_loaded_once_without_runtime_downloads(self) -> None:
        instructions_util = ifbench_instructions.instructions_util
        instructions_util._english_stopwords.cache_clear()
        words = mock.Mock(return_value=["the", "and"])
        with (
            mock.patch.object(instructions_util.nltk.data, "find"),
            mock.patch.object(
                instructions_util.nltk.corpus,
                "stopwords",
                SimpleNamespace(words=words),
            ),
            mock.patch.object(instructions_util.nltk, "download") as nltk_download,
        ):
            self.assertEqual(instructions_util.count_stopwords("the fox and the dog"), 3)
            self.assertEqual(instructions_util.count_stopwords("and then"), 1)
        instructions_util._english_stopwords.cache_clear()

        words.assert_called_once_with("english")
        nltk_download.assert_not_called()

    def test_database_checkpoint_path_is_not_bypassed_by_lighteval_file_cache(self) -> None:
        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig

        config = LiteLLMModelConfig.from_args("model_name=openai/test,use_cache=false")
        client = LiteLLMClient(config)

        self.assertFalse(config.use_cache)
        self.assertIsNone(client._cache)
        self.assertTrue(hasattr(lighteval_raw_completion.greedy_until, "__wrapped__"))

    def test_open_think_prefill_continuation_is_not_answer_text(self) -> None:
        self.assertEqual(
            lighteval_raw_completion._strip_prefill_continuation(
                ">  def solve():\n    return 1", "User: {query}\nAssistant: <think></think"
            ),
            "def solve():\n    return 1",
        )
        self.assertEqual(
            lighteval_raw_completion._strip_prefill_continuation(">reasoning", "Assistant: <think"),
            "reasoning",
        )
        self.assertEqual(
            lighteval_raw_completion._strip_prefill_continuation("> legitimate", "Assistant:"),
            "> legitimate",
        )

    def test_raw_request_preserves_finish_reason_usage_and_unprocessed_text(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"index": 0, "text": "> def solve():\n    return 1", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:8000/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        with mock.patch.dict(os.environ, {"HELICOPTER_PROMPT_TEMPLATE": "Assistant: <think></think"}, clear=False):
            with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value={}):
                with mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response) as post:
                    result = lighteval_raw_completion._request(client, "prompt", 32, 1, None)

        self.assertEqual(result.text, ["def solve():\n    return 1"])
        self.assertEqual(result.raw_text, ["> def solve():\n    return 1"])
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.usage["total_tokens"], 14)
        self.assertEqual(post.call_args.kwargs["timeout"], 10)

    def test_rendered_prompt_reaches_completion_payload_with_real_newline(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:19315/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        prompt = lighteval_raw_completion._render_prompt(
            "User: {query}\nAssistant: <think></think",
            "raw question",
        )
        with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value={}):
            with mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response) as post:
                lighteval_raw_completion._request(client, prompt, 32, 1, None)

        sent_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(sent_prompt, "User: raw question\nAssistant: <think></think")
        self.assertIn(b"\x0a", sent_prompt.encode("utf-8"))
        self.assertNotIn(b"\\n", sent_prompt.encode("utf-8"))

    def test_chat_request_uses_official_problem_and_fake_think_mode(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ">answer"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:19315/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        with mock.patch.dict(
            os.environ,
            {
                "HELICOPTER_PROMPT_MODE": "naive_nocot",
                "HELICOPTER_PROMPT_TEMPLATE": "User: {query}\n\nAssistant: <think></think",
            },
            clear=False,
        ):
            with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value={}):
                with mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response) as post:
                    result = lighteval_raw_completion._request(
                        client,
                        "User: ignored wrapper",
                        32,
                        1,
                        ["\nUser:"],
                        problem="official instruction plus question",
                    )

        self.assertEqual(result.text, ["answer"])
        self.assertEqual(
            post.call_args.args[0],
            "http://127.0.0.1:19315/v1/chat/completions",
        )
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("prompt", payload)
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "official instruction plus question"}],
        )
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"rwkv_generation_prompt": "fake_think"},
        )

    def test_chat_request_uses_request_template_for_normal_mode(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:19315/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_PROMPT_MODE": "normal_cot"},
            clear=False,
        ):
            with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value={}):
                with mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response) as post:
                    lighteval_raw_completion._request(
                        client,
                        "ignored wrapper",
                        32,
                        1,
                        ["\nUser:"],
                        problem="official instruction plus question",
                    )

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertIn("chat_template", payload)
        self.assertIn("User✿{{ message['content'] }}✿", payload["chat_template"])
        self.assertIn("Bot✿<think", payload["chat_template"])

    def test_raw_request_forwards_every_configured_vllm_sampling_field(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"index": 0, "text": "answer", "finish_reason": "stop"}],
            "usage": {},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:29573/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        overrides = {
            "temperature": 0.96,
            "top_p": 0.76,
            "top_k": 32,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.1,
            "penalty_decay": 0.988,
        }
        with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value=overrides), \
             mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response) as post:
            lighteval_raw_completion._request(client, "prompt", 32, 1, None)

        payload = post.call_args.kwargs["json"]
        for key, value in overrides.items():
            self.assertEqual(payload[key], value)

    def test_raw_request_preserves_completion_endpoint_error_body(self) -> None:
        response = mock.Mock()
        response.status_code = 400
        response.text = '{"error":{"message":"request rejected"}}'
        response.raise_for_status.side_effect = lighteval_raw_completion.requests.HTTPError(
            "400 Client Error"
        )
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:29573/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )

        with mock.patch.object(lighteval_raw_completion, "load_sampling_overrides", return_value={}), \
             mock.patch.object(lighteval_raw_completion.requests, "post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                'HTTP 400: {"error":{"message":"request rejected"}}',
            ):
                lighteval_raw_completion._request(client, "prompt", 32, 1, None)

    def test_context_budget_uses_endpoint_token_count_only_for_tight_prompts(self) -> None:
        client = SimpleNamespace(
            model="openai/rwkv-test",
            base_url="http://127.0.0.1:19315/v1",
            api_key="key",
            timeout=10,
            max_length=10240,
        )
        with mock.patch.object(lighteval_raw_completion.requests, "post") as post:
            self.assertEqual(
                lighteval_raw_completion._fit_request_to_context(
                    client,
                    prompt="short prompt",
                    requested_max_tokens=1280,
                ).max_tokens,
                1280,
            )
            post.assert_not_called()

        tokenized = mock.Mock()
        tokenized.raise_for_status.return_value = None
        tokenized.json.return_value = {"count": 8961, "max_model_len": 10240}
        with mock.patch.object(
            lighteval_raw_completion.requests,
            "post",
            return_value=tokenized,
        ) as post:
            fit = lighteval_raw_completion._fit_request_to_context(
                client,
                prompt="x" * 9000,
                requested_max_tokens=1280,
            )

        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:19315/tokenize")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "rwkv-test")
        self.assertEqual(fit.max_tokens, 1279)
        self.assertIsNone(fit.truncate_prompt_tokens)
        self.assertEqual(fit.truncated_prompt_tokens, 0)

    def test_forced_context_budget_cannot_be_overridden_by_task_sampling(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"index": 0, "text": "answer", "finish_reason": "stop"}],
            "usage": {},
        }
        client = SimpleNamespace(
            model="openai/model",
            base_url="http://127.0.0.1:29573/v1",
            api_key="key",
            timeout=10,
            API_MAX_RETRY=1,
            API_RETRY_SLEEP=0,
            API_RETRY_MULTIPLIER=1,
        )
        policy = {
            "tasks": {
                "ifbench_multiturn": {
                    "sampling": {"max_tokens": 1280, "temperature": 0.96},
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ), mock.patch.object(
            lighteval_raw_completion,
            "load_sampling_overrides",
            return_value={"max_tokens": 1280},
        ), mock.patch.object(
            lighteval_raw_completion.requests,
            "post",
            return_value=response,
        ) as post:
            lighteval_raw_completion._request(
                client,
                "prompt",
                1279,
                1,
                None,
                force_max_tokens=True,
                truncate_prompt_tokens=8960,
                task_name="g1h__ifbench_multiturn|0",
            )

        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 1279)
        self.assertEqual(post.call_args.kwargs["json"]["truncate_prompt_tokens"], 8960)
        self.assertEqual(post.call_args.kwargs["json"]["truncation_side"], "left")

    def test_scoreboard_keeps_full_completion_and_extracted_eval_answer_separate(self) -> None:
        response = ModelResponse(text=["Reasoning. Answer: B"], text_post_processed=[" B"])
        response.finish_reason = "stop"
        response.raw_text = ["> Reasoning. Answer: B"]

        payload = scoreboard_bridge._response_payload(response)

        self.assertEqual(scoreboard_bridge._completion_answer(payload), "Reasoning. Answer: B")
        self.assertEqual(scoreboard_bridge._answer(payload), " B")
        self.assertEqual(scoreboard_bridge._stop_reason(payload), "stop")
        self.assertEqual(payload["raw_text"], ["> Reasoning. Answer: B"])

    def test_lighteval_rollout_index_keeps_post_processed_answer(self) -> None:
        response = ModelResponse(
            text=["reasoning", "reasoning 2"],
            text_post_processed=[" A", " B"],
            reasonings=["r1", "r2"],
        )
        indexed = response[1]
        self.assertEqual(indexed.text, ["reasoning 2"])
        self.assertEqual(indexed.text_post_processed, [" B"])
        self.assertEqual(indexed.reasonings, ["r2"])

    def test_lighteval_checkpoint_session_reuses_one_db_lifecycle(self) -> None:
        store = SimpleNamespace(
            insert_completion_payloads_with_task=mock.AsyncMock(
                side_effect=[("7", 1), ("7", 1)]
            ),
        )
        response = ModelResponse(text=["answer"])
        response.raw_text = [">answer"]
        response.finish_reason = "stop"
        response.usage = {"total_tokens": 2}

        with mock.patch(
            "scoreboard_server.db.settings.DatabaseSettings.from_env",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "scoreboard_server.db.connection.init_db",
            new=mock.AsyncMock(),
        ) as init_db, mock.patch(
            "scoreboard_server.db.connection.close_db",
            new=mock.AsyncMock(),
        ) as close_db, mock.patch(
            "scoreboard_server.db.repository.ScoreboardStore",
            return_value=store,
        ):
            with scoreboard_bridge.LightevalCheckpointSession(
                task_id="7",
                dataset="gsm8k",
                num_samples=2,
            ) as session:
                for sample_index in range(2):
                    session.checkpoint(
                        task_name="g1h__gsm8k|0",
                        sample_index=sample_index,
                        doc=SimpleNamespace(id=str(sample_index), task_name="g1h__gsm8k|0"),
                        response=response,
                        repeat_indices=[0],
                        generation_size=32,
                    )

        init_db.assert_awaited_once()
        close_db.assert_awaited_once()
        self.assertEqual(store.insert_completion_payloads_with_task.await_count, 2)

    def test_lighteval_checkpoint_session_inserts_each_rollout_individually(self) -> None:
        store = SimpleNamespace(
            insert_completion_payloads_with_task=mock.AsyncMock(
                side_effect=[("7", 1), ("7", 1)]
            ),
        )
        response = ModelResponse(text=["first", "second"])
        response.raw_text = ["first", "second"]
        response.finish_reason = ["stop", "stop"]

        with mock.patch(
            "scoreboard_server.db.settings.DatabaseSettings.from_env",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "scoreboard_server.db.connection.init_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.connection.close_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.repository.ScoreboardStore",
            return_value=store,
        ):
            with scoreboard_bridge.LightevalCheckpointSession(
                task_id="7",
                dataset="gsm8k",
                num_samples=2,
            ) as session:
                session.checkpoint(
                    task_name="g1h__gsm8k|0",
                    sample_index=0,
                    doc=SimpleNamespace(id="0", task_name="g1h__gsm8k|0"),
                    response=response,
                    repeat_indices=[0, 1],
                    generation_size=32,
                )

        self.assertEqual(store.insert_completion_payloads_with_task.await_count, 2)
        calls = store.insert_completion_payloads_with_task.await_args_list
        self.assertEqual([len(call.kwargs["payloads"]) for call in calls], [1, 1])

    def test_lighteval_task_is_not_created_before_first_completion(self) -> None:
        store = SimpleNamespace(
            get_resume_context=mock.AsyncMock(
                return_value=SimpleNamespace(can_resume=False, task_id=None)
            ),
            get_or_create_task=mock.AsyncMock(),
        )
        with mock.patch(
            "scoreboard_server.db.settings.DatabaseSettings.from_env",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "scoreboard_server.db.connection.init_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.connection.close_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.repository.ScoreboardStore",
            return_value=store,
        ):
            task_id = scoreboard_bridge.asyncio.run(
                scoreboard_bridge._prepare_lighteval_task(
                    model="rwkv-test",
                    dataset="gsm8k",
                )
            )

        self.assertEqual(task_id, "")
        store.get_or_create_task.assert_not_awaited()

    def test_first_completion_atomically_creates_lighteval_task(self) -> None:
        store = SimpleNamespace(
            insert_completion_payloads_with_task=mock.AsyncMock(
                side_effect=[("19", 1), ("19", 1)]
            ),
        )
        response = ModelResponse(text=["answer"])
        response.raw_text = ["answer"]
        response.finish_reason = "stop"

        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_SCOREBOARD_MODEL_NAME": "rwkv-test"},
        ), mock.patch(
            "scoreboard_server.db.settings.DatabaseSettings.from_env",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "scoreboard_server.db.connection.init_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.connection.close_db",
            new=mock.AsyncMock(),
        ), mock.patch(
            "scoreboard_server.db.repository.ScoreboardStore",
            return_value=store,
        ):
            with scoreboard_bridge.LightevalCheckpointSession(
                task_id=None,
                dataset="gsm8k",
                num_samples=2,
            ) as session:
                for sample_index in range(2):
                    session.checkpoint(
                        task_name="g1h__gsm8k|0",
                        sample_index=sample_index,
                        doc=SimpleNamespace(id=str(sample_index), task_name="g1h__gsm8k|0"),
                        response=response,
                        repeat_indices=[0],
                        generation_size=32,
                    )
                self.assertEqual(session.task_id, "19")
                self.assertEqual(os.environ["HELICOPTER_SCOREBOARD_TASK_ID"], "19")

        self.assertEqual(store.insert_completion_payloads_with_task.await_count, 2)

    def test_generation_checkpoint_first_write_is_completed_result(self) -> None:
        response = ModelResponse(text=["final answer"])
        response.raw_text = ["final answer"]
        response.finish_reason = "stop"
        response.usage = {"total_tokens": 9}

        payload = scoreboard_bridge._generation_payloads(
            task_name="g1h__ifbench_multiturn|0",
            sample_index=247,
            doc={"id": "raw-row-1309", "task_name": "g1h__ifbench_multiturn|0"},
            response=response,
            repeat_indices=[0],
            generation_size=1280,
        )[0]

        self.assertEqual(payload["status"], "Completed")
        self.assertEqual(payload["sample_index"], 247)
        self.assertEqual(payload["task_id"], "raw-row-1309")
        self.assertEqual(payload["completion1"], "final answer")
        self.assertEqual(payload["agent_result"]["doc"]["id"], "raw-row-1309")
        self.assertTrue(payload["stats"]["generation_checkpoint"])
        self.assertEqual(payload["sampling_config"]["effective_generation_size"], 1280)
        self.assertFalse(hasattr(scoreboard_bridge.LightevalCheckpointSession, "register_pending"))

    def test_resume_rejects_lighteval_index_to_dataset_row_mismatch(self) -> None:
        checkpoint = {
            0: {
                "model_response": {"text": ["stale answer"]},
                "dataset_row_id": 1309,
                "prompt": "User: old row\nAssistant: <think></think",
            }
        }

        with self.assertRaisesRegex(RuntimeError, "not current row 1386"):
            lighteval_raw_completion._validated_stored_rollouts(
                checkpoint,
                sample_index=1386,
                doc={"id": 1386},
                prompt="User: new row\nAssistant: <think></think",
            )

    def test_resume_accepts_only_matching_dataset_row_and_prompt(self) -> None:
        model_response = {"text": ["answer"]}
        checkpoint = {
            0: {
                "model_response": model_response,
                "dataset_row_id": 1309,
                "prompt": "User: exact row\nAssistant: <think></think",
            }
        }

        restored = lighteval_raw_completion._validated_stored_rollouts(
            checkpoint,
            sample_index=1386,
            doc={"id": 1309},
            prompt="User: exact row\nAssistant: <think></think",
        )

        self.assertEqual(restored, {0: model_response})

    def test_database_scoring_rejects_sample_index_dataset_row_mismatch(self) -> None:
        pipeline = SimpleNamespace(
            sampling_docs={
                lighteval_db_pipeline.SamplingMethod.GENERATIVE: [
                    SimpleNamespace(id="current-row-1386", num_samples=1)
                ]
            }
        )
        stored = [{
            "sample_index": 0,
            "repeat_index": 0,
            "status": "Completed",
            "context": {
                "stats": {"dataset_row_id": "raw-row-1309"},
                "agent_result": {
                    "doc": {"id": "raw-row-1309"},
                    "model_response": {"text": ["stale answer"]},
                },
            },
        }]

        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_SCOREBOARD_TASK_ID": "42"},
            clear=False,
        ), mock.patch.object(
            lighteval_db_pipeline,
            "load_lighteval_generation",
            return_value=stored,
        ), self.assertRaisesRegex(RuntimeError, "not current row 'current-row-1386'"):
            lighteval_db_pipeline._responses_from_database(pipeline)

    def test_task_request_policy_combines_native_and_toml_stops(self) -> None:
        policy = {
            "tasks": {
                "gsm8k": {
                    "domain": "math",
                    "inherit_task_stops": True,
                    "stop": ["\\nUser:"],
                    "sampling": {"temperature": 0.96},
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ):
            self.assertEqual(
                lighteval_raw_completion._configured_stops(
                    "g1h__gsm8k|0", ["Question:"]
                ),
                ["Question:", "\\nUser:"],
            )
            self.assertEqual(
                lighteval_raw_completion._configured_sampling("g1h__gsm8k|0"),
                {"temperature": 0.96},
            )

    def test_task_request_policy_is_inherited_by_lighteval_subtasks(self) -> None:
        policy = {
            "tasks": {
                "mmlu": {
                    "domain": "knowledge",
                    "sampling": {"max_tokens": 8192},
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ):
            self.assertEqual(
                lighteval_raw_completion._configured_sampling(
                    "g1h__mmlu:machine_learning|0"
                ),
                {"max_tokens": 8192},
            )

    def test_request_policy_combines_domain_and_format_stops(self) -> None:
        policy = commands.resolve_lighteval_task_request_policy(
            config={
                "prompt": {
                    "template": "User: {query}\nAssistant:",
                    "formats": {
                        "python_program": {
                            "task_prefixes": ["lcb:"],
                            "template": "User: {query}\nAssistant: ```python",
                        }
                    },
                },
                "stops": {
                    "inherit_task_stops": False,
                    "domains": {"coding": ["\nUser:"]},
                    "formats": {"python_program": ["\n```"]},
                },
                "sampling": {},
            },
            selected_tasks=["lcb:codegeneration"],
            base_sampling={},
        )
        self.assertEqual(
            policy["tasks"]["lcb:codegeneration"]["stop"],
            ["\nUser:", "\n```"],
        )

    def test_task_request_policy_can_use_only_toml_flower_stop(self) -> None:
        policy = {
            "tasks": {
                "ifeval": {
                    "domain": "instruction_following",
                    "inherit_task_stops": False,
                    "stop": ["✿"],
                    "sampling": {},
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ):
            self.assertEqual(
                lighteval_raw_completion._configured_stops("g1h__ifeval|0", []),
                ["✿"],
            )

    def test_task_request_policy_uses_domain_prompt_template(self) -> None:
        policy = {
            "tasks": {
                "arc:challenge": {
                    "domain": "knowledge",
                    "prompt_template": "User: {query}\nFinal answer only.\nAssistant: <think",
                    "sampling": {},
                }
            }
        }
        with mock.patch.dict(
            os.environ,
            {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
            clear=False,
        ):
            self.assertEqual(
                lighteval_raw_completion._configured_prompt_template(
                    "g1h__arc:challenge|0", "User: {query}\nAssistant:"
                ),
                "User: {query}\nFinal answer only.\nAssistant: <think",
            )

    def test_multiturn_generation_persists_two_stages_per_rollout(self) -> None:
        first = ModelResponse(text=["first-a", "first-b"], input="p1")
        first.raw_text = [">first-a", ">first-b"]
        first.finish_reason = ["stop", "stop"]
        first.usage = [{"total_tokens": 10}, {"total_tokens": 11}]
        second_a = ModelResponse(text=["second-a"], input="p2a")
        second_a.raw_text = [">second-a"]
        second_a.finish_reason = ["stop"]
        second_a.usage = {"total_tokens": 12}
        second_b = ModelResponse(text=["second-b"], input="p2b")
        second_b.raw_text = [">second-b"]
        second_b.finish_reason = ["length"]
        second_b.usage = {"total_tokens": 13}
        fit = lighteval_raw_completion._RequestContextFit(128, None, 10, 10240)
        policy = {
            "tasks": {
                "mt_bench": {
                    "multi_turn_template": (
                        "{first_prompt}{first_answer}\nUser: {query}\n"
                        "Assistant: <think></think"
                    ),
                    "sampling": {},
                }
            }
        }
        doc = SimpleNamespace(
            choices=[],
            specific={"multi_turn_queries": ["first question", "second question"]},
        )
        with (
            mock.patch.dict(
                os.environ,
                {"HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(policy)},
                clear=False,
            ),
            mock.patch.object(
                lighteval_raw_completion,
                "_request",
                side_effect=[first, second_a, second_b],
            ) as request,
            mock.patch.object(
                lighteval_raw_completion,
                "_fit_request_to_context",
                return_value=fit,
            ),
        ):
            response = lighteval_raw_completion._generate_two_turn_response(
                SimpleNamespace(),
                doc=doc,
                task_name="g1h__mt_bench|0",
                first_prompt="User: first question\nAssistant: <think></think",
                queries=["first question", "second question"],
                max_tokens=128,
                missing_count=2,
                stops=["\nUser:"],
                first_template="User: {query}\nAssistant: <think></think",
                first_context_fit=fit,
            )

        self.assertEqual(response.text, ["second-a", "second-b"])
        self.assertEqual(len(response.stages_by_rollout), 2)
        self.assertEqual(
            response.stages_by_rollout[0][0]["completion"],
            "first-a",
        )
        self.assertEqual(
            response.stages_by_rollout[0][1]["completion"],
            "second-a",
        )
        self.assertIn("first-a\nUser: second question", request.call_args_list[1].args[1])
        self.assertIn("first-b\nUser: second question", request.call_args_list[2].args[1])


    def test_sampling_identity_omits_absent_multiturn_template(self) -> None:
        base_policy = {
            "tasks": {
                "simpleqa": {
                    "domain": "knowledge",
                    "prompt_template": "User: {query}\nAssistant:",
                    "inherit_task_stops": False,
                    "stop": ["\nUser:"],
                    "sampling": {"max_tokens": 512},
                }
            }
        }
        values = {
            "HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(base_policy),
        }
        sampling = scoreboard_bridge.sampling_config_from_env(values)
        persisted = sampling["task_request_policy"]
        self.assertNotIn("multi_turn_template", persisted)

        base_policy["tasks"]["simpleqa"]["multi_turn_template"] = (
            "{first_prompt}{first_answer}\nUser: {query}\nAssistant:"
        )
        values["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"] = json.dumps(base_policy)
        sampling = scoreboard_bridge.sampling_config_from_env(values)
        self.assertEqual(
            sampling["task_request_policy"]["multi_turn_template"],
            "{first_prompt}{first_answer}\nUser: {query}\nAssistant:",
        )

    def test_scoreboard_expands_two_stage_completion_metadata(self) -> None:
        response = {
            "finish_reason": "stop",
            "stages": [
                {"prompt": "p1", "completion": "c1", "stop_reason": "length"},
                {"prompt": "p2", "completion": "c2", "stop_reason": "stop"},
            ],
        }
        self.assertEqual(
            scoreboard_bridge._completion_stages(
                response, fallback_prompt="fallback", fallback_completion="fallback"
            ),
            {
                "prompt1": "p1",
                "completion1": "c1",
                "stop_reason1": "length",
                "prompt2": "p2",
                "completion2": "c2",
                "stop_reason2": "stop",
            },
        )

    def test_scoreboard_expands_each_rollout(self) -> None:
        response = {
            "text": ["first", "second", "third", "fourth"],
            "text_post_processed": ["A", "B", "C", "D"],
            "raw_text": [">first", ">second", ">third", ">fourth"],
            "finish_reason": ["stop", "length", "stop", "stop"],
            "usage": [
                [{"total_tokens": 1}, {"total_tokens": 2}],
                [{"total_tokens": 3}, {"total_tokens": 4}],
            ],
        }
        self.assertEqual(scoreboard_bridge._rollout_count(response), 4)
        rollout = scoreboard_bridge._rollout_response(response, 1)
        self.assertEqual(scoreboard_bridge._completion_answer(rollout), "second")
        self.assertEqual(scoreboard_bridge._answer(rollout), "B")
        self.assertEqual(scoreboard_bridge._stop_reason(rollout), "length")
        self.assertEqual(rollout["raw_text"], ">second")
        self.assertEqual(rollout["usage"], {"total_tokens": 4})

    def test_scoreboard_recovers_completion_from_durable_stages(self) -> None:
        response = {
            "text": "",
            "text_post_processed": "",
            "stages": [
                {"prompt": "p", "completion": "raw completion", "stop_reason": "stop"},
            ],
        }
        self.assertEqual(scoreboard_bridge._completion_answer(response), "raw completion")

        self.assertEqual(scoreboard_bridge._dataset("g1h__human_eval|0"), "human_eval")


    def test_deep_instruction_metadata_is_not_truncated(self) -> None:
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        from scoreboard_server.cores.normalize import sanitize_json

        value: object = "leaf"
        for _ in range(12):
            value = {"nested": value}
        sanitized = sanitize_json(value)
        self.assertNotIn("[truncated depth]", json.dumps(sanitized))

class ParserTests(unittest.TestCase):
    def test_agent_harness_parser_accepts_plan_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "agent-harness",
                "plan",
                "swe_bench_verified",
                "--model",
                "g1d-0.4b",
                "--output-dir",
                "tmp/agent",
            ]
        )

        self.assertEqual(args.eval_command, "agent-harness")
        self.assertEqual(args.agent_action, "plan")
        self.assertEqual(args.benchmark, "swe_bench_verified")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.output_dir, "tmp/agent")

    def test_agent_harness_parser_accepts_run_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "agent-harness",
                "run",
                "browsecomp",
                "--model",
                "g1d-0.4b",
                "--base-url",
                "http://127.0.0.1:8000/v1",
                "--max-samples",
                "1",
                "--no-server",
                "--allow-proxy",
            ]
        )

        self.assertEqual(args.agent_action, "run")
        self.assertEqual(args.benchmark, "browsecomp")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.max_samples, 1)
        self.assertTrue(args.no_server)
        self.assertTrue(args.allow_proxy)

    def test_agent_harness_parser_accepts_convert_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "agent-harness",
                "convert",
                "swe_bench_verified",
                "--input",
                "tmp/raw.jsonl",
                "--output",
                "tmp/predictions.jsonl",
                "--model",
                "g1d-0.4b",
            ]
        )

        self.assertEqual(args.agent_action, "convert")
        self.assertEqual(args.benchmark, "swe_bench_verified")
        self.assertEqual(args.input, "tmp/raw.jsonl")
        self.assertEqual(args.output, "tmp/predictions.jsonl")
        self.assertEqual(args.target, "auto")

    def test_function_calling_parser_keeps_small_lighteval_like_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "function-calling",
                "g1d-0.4b",
                "bfcl_v3",
                "--max-samples",
                "2",
                "--output-dir",
                "tmp/fc",
                "--scoreboard",
            ]
        )

        self.assertEqual(args.eval_command, "function-calling")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.tasks, "bfcl_v3")
        self.assertEqual(args.max_samples, 2)
        self.assertEqual(args.output_dir, "tmp/fc")
        self.assertTrue(args.scoreboard)
        self.assertFalse(hasattr(args, "wkv_mode"))
        self.assertFalse(hasattr(args, "tensor_parallel_size"))

    def test_function_calling_parser_rejects_vllm_detail_flags(self) -> None:
        parser = helicopter_main.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["eval", "function-calling", "g1d-0.4b", "bfcl_v3", "--wkv-mode", "fp16"])

    def test_eval_perf_parser_accepts_raw_completions_surface(self) -> None:
        parser = helicopter_main.build_parser()
        args = parser.parse_args(
            [
                "eval",
                "perf",
                "g1d-0.4b",
                "--profile",
                "prefill",
                "--prompt-tokens",
                "2048",
                "--output-tokens",
                "8",
                "--requests",
                "4",
                "--concurrency",
                "2",
                "--ignore-eos",
            ]
        )

        self.assertEqual(args.eval_command, "perf")
        self.assertEqual(args.model, "g1d-0.4b")
        self.assertEqual(args.profile, "prefill")
        self.assertEqual(args.prompt_tokens, 2048)
        self.assertEqual(args.output_tokens, 8)
        self.assertEqual(args.requests, 4)
        self.assertEqual(args.concurrency, 2)
        self.assertTrue(args.ignore_eos)


class FunctionCallingTests(unittest.TestCase):
    def test_openai_tool_calls_are_normalized_and_scored(self) -> None:
        sample = function_calling.FunctionCallingSample(
            task_name="bfcl_exec_parallel",
            sample_id="unit",
            kind="bfcl",
            messages=[],
            tools=[],
            specific={
                "expected_calls_json": json.dumps(
                    [
                        {"name": "calc", "arguments": {"x": 1}},
                        {"name": "lookup", "arguments": {"key": "a"}},
                    ]
                )
            },
        )
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{\"key\":\"a\"}"},
                            },
                            {
                                "type": "function",
                                "function": {"name": "calc", "arguments": "{\"x\":1}"},
                            },
                        ]
                    }
                }
            ]
        }

        calls = function_calling.tool_calls_from_response(response)

        self.assertEqual(
            calls,
            [
                {"name": "lookup", "arguments": {"key": "a"}},
                {"name": "calc", "arguments": {"x": 1}},
            ],
        )
        self.assertEqual(function_calling.score_calls(sample, calls), 1.0)

    @staticmethod
    def _pipeline_sample(sample_id: str) -> function_calling.FunctionCallingSample:
        return function_calling.FunctionCallingSample(
            task_name="bfcl_v3",
            sample_id=sample_id,
            kind="bfcl",
            messages=[],
            tools=[],
            specific={"expected_calls_json": json.dumps([{"name": "calc", "arguments": {"x": 1}}])},
        )

    @staticmethod
    def _pipeline_args() -> Namespace:
        return Namespace(
            model="g1d-0.4b",
            tasks="bfcl_v3",
            dry_run=False,
            base_url="http://127.0.0.1:29573/v1",
            output_dir=None,
            max_samples=None,
            no_server=True,
            keep_server=False,
            config="configs/naive-nocot.toml",
        )

    def test_function_calling_score_uses_database_without_model(self) -> None:
        samples = [self._pipeline_sample("0"), self._pipeline_sample("1")]

        def stored(index: int) -> dict[str, object]:
            return {
                "sample_index": index,
                "status": "Completed",
                "context": {"agent_result": {"run_result": {
                    "task_name": "bfcl_v3",
                    "sample_id": str(index),
                    "score": 0.0,
                    "actual_calls": [{"name": "calc", "arguments": {"x": 1}}],
                    "raw_response": {"choices": []},
                    "error": None,
                }}},
            }

        with mock.patch.object(function_calling, "load_samples", return_value=samples), \
             mock.patch.object(function_calling, "prepare_function_calling_task", return_value="42"), \
             mock.patch.object(function_calling, "load_function_calling_generation", return_value=[stored(0), stored(1)]), \
             mock.patch.object(function_calling, "run_samples") as run_samples, \
             mock.patch.object(function_calling, "write_function_calling_results", return_value=["42"]) as write_results:
            exit_code = function_calling.run_function_calling_eval(
                self._pipeline_args(),
                root=ROOT,
                env={"HELICOPTER_PIPELINE_STAGE": "score"},
                config=load_example_config(),
            )

        self.assertEqual(exit_code, 0)
        run_samples.assert_not_called()
        self.assertEqual(write_results.call_args.kwargs["task_id"], "42")
        scored = write_results.call_args.kwargs["results"]
        self.assertEqual([result.score for result in scored], [1.0, 1.0])

    def test_function_calling_generate_resumes_only_missing_samples(self) -> None:
        samples = [self._pipeline_sample("0"), self._pipeline_sample("1")]
        completed = {
            "sample_index": 0,
            "status": "Completed",
            "context": {"agent_result": {"run_result": {"sample_id": "0"}}},
        }
        generated = function_calling.FunctionCallingRunResult(
            task_name="bfcl_v3",
            sample_id="1",
            score=1.0,
            actual_calls=[{"name": "calc", "arguments": {"x": 1}}],
            raw_response={"choices": []},
        )

        def fake_run(selected, *, on_result, **kwargs):
            self.assertEqual([sample.sample_id for sample in selected], ["1"])
            on_result(0, generated)
            return [generated]

        with mock.patch.object(function_calling, "load_samples", return_value=samples), \
             mock.patch.object(function_calling, "prepare_function_calling_task", return_value="42"), \
             mock.patch.object(function_calling, "load_function_calling_generation", return_value=[completed]), \
             mock.patch.object(function_calling, "run_samples", side_effect=fake_run), \
             mock.patch.object(function_calling, "checkpoint_function_calling_result") as checkpoint, \
             mock.patch.object(function_calling, "write_function_calling_results") as write_results:
            exit_code = function_calling.run_function_calling_eval(
                self._pipeline_args(),
                root=ROOT,
                env={"HELICOPTER_PIPELINE_STAGE": "generate"},
                config=load_example_config(),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(checkpoint.call_count, 1)
        self.assertEqual(checkpoint.call_args.kwargs["sample_index"], 1)
        write_results.assert_not_called()

    def test_function_calling_generate_binds_task_on_first_real_write(self) -> None:
        samples = [self._pipeline_sample("0"), self._pipeline_sample("1")]
        generated = [
            function_calling.FunctionCallingRunResult(
                task_name="bfcl_v3",
                sample_id=str(index),
                score=1.0,
                actual_calls=[{"name": "calc", "arguments": {"x": 1}}],
                raw_response={"choices": []},
            )
            for index in range(2)
        ]

        def fake_run(selected, *, on_result, **kwargs):
            self.assertEqual(selected, samples)
            for index, result in enumerate(generated):
                on_result(index, result)
            return generated

        with mock.patch.object(function_calling, "load_samples", return_value=samples), \
             mock.patch.object(function_calling, "prepare_function_calling_task", return_value=""), \
             mock.patch.object(function_calling, "load_function_calling_generation") as load_generation, \
             mock.patch.object(function_calling, "run_samples", side_effect=fake_run), \
             mock.patch.object(
                 function_calling,
                 "checkpoint_function_calling_result",
                 side_effect=["42", "42"],
             ) as checkpoint:
            exit_code = function_calling.run_function_calling_eval(
                self._pipeline_args(),
                root=ROOT,
                env={"HELICOPTER_PIPELINE_STAGE": "generate"},
                config=load_example_config(),
            )

        self.assertEqual(exit_code, 0)
        load_generation.assert_not_called()
        self.assertEqual(
            [call.kwargs["task_id"] for call in checkpoint.call_args_list],
            ["", "42"],
        )
        self.assertTrue(all(call.kwargs["model"] == "g1d-0.4b" for call in checkpoint.call_args_list))
        self.assertTrue(
            all(
                call.kwargs["config_path"] == "configs/naive-nocot.toml"
                for call in checkpoint.call_args_list
            )
        )

    def test_function_calling_dry_run_does_not_load_samples(self) -> None:
        args = Namespace(
            model="g1d-0.4b",
            tasks="bfcl_v3",
            dry_run=True,
            base_url=None,
            output_dir=None,
            max_samples=2,
            no_server=False,
            keep_server=False,
            scoreboard=False,
        )
        with mock.patch.object(function_calling, "load_samples") as load_samples:
            with mock.patch("sys.stdout", new=io.StringIO()):
                exit_code = function_calling.run_function_calling_eval(
                    args,
                    root=ROOT,
                    env={},
                    config=load_example_config(),
                )

        self.assertEqual(exit_code, 0)
        load_samples.assert_not_called()

    def test_function_calling_infer_namespace_forwards_batch_server_overrides(self) -> None:
        args = Namespace(
            model="g1d-0.4b",
            dry_run=False,
            wkv_mode="fp32io16",
            emb_device="gpu",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.72,
            max_num_seqs=128,
            max_num_batched_tokens=32768,
            enable_auto_tool_choice=None,
            vllm_env=["VLLM_USE_RAPID_SAMPLER=1"],
            _config=load_example_config(),
        )

        namespace = function_calling.infer_args_namespace(args, port="8012")

        self.assertEqual(namespace.port, "8012")
        self.assertEqual(namespace.served_model_name, "g1d-0.4b")
        self.assertEqual(namespace.wkv_mode, "fp32io16")
        self.assertEqual(namespace.emb_device, "gpu")
        self.assertEqual(namespace.tensor_parallel_size, 2)
        self.assertEqual(namespace.gpu_memory_utilization, 0.72)
        self.assertEqual(namespace.max_num_seqs, 128)
        self.assertEqual(namespace.max_num_batched_tokens, 32768)
        self.assertTrue(namespace.enable_auto_tool_choice)
        self.assertEqual(namespace.vllm_env, ["VLLM_USE_RAPID_SAMPLER=1"])

    def test_function_calling_aggregates_latency_and_usage(self) -> None:
        results = [
            function_calling.FunctionCallingRunResult(
                "bfcl_v3",
                "1",
                1.0,
                [{"name": "lookup", "arguments": {}}],
                None,
                elapsed_seconds=0.10,
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
            function_calling.FunctionCallingRunResult(
                "bfcl_v3",
                "2",
                0.0,
                [],
                None,
                error="timeout",
                elapsed_seconds=0.30,
                prompt_tokens=20,
                completion_tokens=4,
                total_tokens=24,
            ),
        ]

        aggregate = function_calling._aggregate_results(results)
        performance = function_calling._aggregate_performance(results, elapsed_seconds=0.5)

        self.assertEqual(aggregate["bfcl_v3"]["prompt_tokens"], 30)
        self.assertEqual(aggregate["bfcl_v3"]["completion_tokens"], 6)
        self.assertEqual(aggregate["bfcl_v3"]["total_tokens"], 36)
        self.assertEqual(aggregate["bfcl_v3"]["e2e_latency_seconds"]["p50"], 0.10)
        self.assertEqual(performance["requests"], 2)
        self.assertEqual(performance["failed_requests"], 1)
        self.assertEqual(performance["tokens_per_second"], 72.0)

    def test_function_calling_loads_local_intermediate_dataset_before_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "helicopter"
            data_path = Path(tmp) / "rwkv-skills" / "data" / "bfcl_simple_python" / "test.jsonl"
            data_path.parent.mkdir(parents=True)
            data_path.write_text(
                json.dumps(
                    {
                        "task_id": "local-1",
                        "instruction": "Call calc with x=1.",
                        "tools": [
                            {
                                "name": "calc",
                                "description": "Calculate.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"x": {"type": "integer"}},
                                    "required": ["x"],
                                },
                            }
                        ],
                        "expected_tool_calls": [{"name": "calc", "arguments": {"x": 1}}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(function_calling.urllib.request, "urlopen", side_effect=AssertionError("network")):
                samples = function_calling.load_samples("bfcl_simple_python", max_samples=1, root=root)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_id, "local-1")
        self.assertEqual(samples[0].kind, "bfcl")
        self.assertEqual(samples[0].tools[0]["type"], "function")
        self.assertEqual(function_calling.score_calls(samples[0], [{"name": "calc", "arguments": {"x": 1}}]), 1.0)

    def test_compact_response_message_keeps_content_and_finish_reason(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "plain fallback",
                        "tool_calls": [],
                    },
                }
            ]
        }

        self.assertEqual(
            function_calling._compact_response_message(response),
            {
                "role": "assistant",
                "content": "plain fallback",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        )


class DotenvTests(unittest.TestCase):
    def test_load_dotenv_supports_simple_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "PLAIN=value",
                        "export EXPORTED=enabled",
                        "QUOTED='space value'",
                        "# ignored",
                        "not-an-assignment",
                    ]
                )
            )

            self.assertEqual(
                env.load_dotenv(env_file),
                {
                    "PLAIN": "value",
                    "EXPORTED": "enabled",
                    "QUOTED": "space value",
                },
            )

    def test_load_env_keeps_command_scoped_environment_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.local").write_text("WEIGHT_PATH=/from-file\n")

            with mock.patch.dict(os.environ, {"WEIGHT_PATH": "/from-env"}, clear=False):
                loaded_env, path = env.load_env(root, ".env.local")

            self.assertEqual(path, root / ".env.local")
            self.assertEqual(loaded_env["WEIGHT_PATH"], "/from-env")


class ConfigResolutionTests(unittest.TestCase):
    def test_default_config_uses_newest_local_toml_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "configs/local"
            local.mkdir(parents=True)
            (root / "configs/example.toml").write_text("")
            (local / "202401010000.toml").write_text("")
            newest = local / "202606290720.toml"
            newest.write_text("")

            self.assertEqual(config.default_config_path(root), newest)

    def test_resolve_model_path_uses_weight_path_directory(self) -> None:
        loaded_config = load_example_config()
        loaded_env = {"WEIGHT_PATH": "/weights/RWKV"}

        model_path, model = config.resolve_model_path(loaded_config, "g1g-1.5b", root=ROOT, env=loaded_env)

        self.assertEqual(model["served_model_name"], "g1g-1.5b")
        self.assertEqual(
            model_path,
            Path("/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth"),
        )


class CommandPlanTests(unittest.TestCase):
    @staticmethod
    def _browsecomp_encrypt(text: str, canary: str) -> str:
        payload = text.encode("utf-8")
        digest = hashlib.sha256(canary.encode("utf-8")).digest()
        key = (digest * ((len(payload) // len(digest)) + 1))[: len(payload)]
        return base64.b64encode(bytes(lhs ^ rhs for lhs, rhs in zip(payload, key))).decode("utf-8")

    def test_infer_plan_uses_vllm_rwkv_contract(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded_config,
        )

        self.assertEqual(
            plan.command[:3],
            ["vllm", "serve", "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth"],
        )
        self.assertEqual(
            command_options(plan.command),
            {
                "--host": "0.0.0.0",
                "--port": "8000",
                "--tokenizer-mode": "rwkv",
                "--load-format": "auto",
                "--served-model-name": "g1g-1.5b",
                "--max-model-len": "8192",
            },
        )
        self.assertEqual(plan.cwd, ROOT)
        self.assertEqual(plan.shown_env, {})
        self.assertEqual({key for key in plan.env if key.startswith("VLLM_")}, set())

    def test_infer_plan_allows_explicit_vllm_env(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(vllm_env=["VLLM_WSL2_ENABLE_PIN_MEMORY=1"]),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_USE_V2_MODEL_RUNNER": "0",
            },
            config=loaded_config,
        )

        self.assertEqual(plan.shown_env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertEqual(plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", plan.env)

    def test_infer_plan_accepts_configured_vllm_env(self) -> None:
        loaded_config = load_example_config()
        loaded_config["infer"] = {
            **loaded_config["infer"],
            "vllm_env": {"VLLM_WSL2_ENABLE_PIN_MEMORY": 1},
        }

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={"WEIGHT_PATH": "/weights/RWKV"},
            config=loaded_config,
        )

        self.assertEqual(plan.shown_env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")
        self.assertEqual(plan.env["VLLM_WSL2_ENABLE_PIN_MEMORY"], "1")

    def test_infer_plan_uses_model_specific_0_4b_runtime(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(model="g1d-0.4b"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        options = command_options(plan.command)
        self.assertEqual(plan.command[0], "/home/chase/GitHub/vllm-rwkv/.venv/bin/vllm")
        self.assertEqual(options["--served-model-name"], "g1d-0.4b")
        self.assertEqual(options["--gpu-memory-utilization"], "0.45")
        self.assertEqual(options["--max-model-len"], "8192")
        self.assertEqual(options["--max-num-seqs"], "8")
        self.assertEqual(options["--max-num-batched-tokens"], "8192")
        self.assertEqual(
            plan.shown_env,
            {
                "VLLM_RWKV7_EMB_DEVICE": "gpu",
                "VLLM_RWKV7_WKV_MODE": "fp16",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_USE_RAPID_SAMPLER": "0",
                "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            },
        )

    def test_lighteval_plan_uses_official_litellm_endpoint(self) -> None:
        loaded_config = load_example_config()
        loaded_config["lighteval"]["request_timeout"] = 900

        plan = commands.build_lighteval_plan(
            lighteval_args(max_samples=3),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:5], ["-m", "lighteval", "endpoint", "litellm"])
        self.assertEqual(plan.command[6], "gsm8k")
        self.assertIn("model_name=openai/g1g-1.5b", plan.command[5])
        self.assertIn("provider=openai", plan.command[5])
        self.assertIn("base_url=http://127.0.0.1:8000/v1", plan.command[5])
        self.assertIn("timeout=900", plan.command[5])
        self.assertIn("max_model_length=8192", plan.command[5])
        self.assertIn("generation_parameters={max_new_tokens:512}", plan.command[5])
        options = command_options(plan.command)
        self.assertEqual(options["--output-dir"], str(ROOT / "results/lighteval"))
        self.assertEqual(options["--max-samples"], "3")
        self.assertEqual(options["--dataset-loading-processes"], "1")
        self.assertEqual(
            options["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_policy_tasks.py"),
        )
        self.assertTrue(options["--load-tasks-multilingual"])
        self.assertTrue(options["--save-details"])
        self.assertEqual(plan.env["OPENAI_API_KEY"], "EMPTY")
        self.assertEqual(plan.env["HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS"], "1")
        self.assertEqual(plan.env["HELICOPTER_PATCH_LIGHTEVAL_DATASET_RETRIES"], "1")
        self.assertEqual(
            plan.env["HELICOPTER_LIGHTEVAL_DATASET_ONLINE_FALLBACK"],
            "1",
        )
        self.assertEqual(plan.env["PYTHONPATH"].split(os.pathsep)[0], str(ROOT / "src/cli"))
        pythonpath = plan.env["PYTHONPATH"].split(os.pathsep)
        self.assertIn(str((ROOT / "src/eval/lighteval/src").resolve()), pythonpath)
        self.assertEqual(
            plan.env["HELICOPTER_LIGHTEEVAL_SOURCE_ROOT"],
            str((ROOT / "src/eval/lighteval/src").resolve()),
        )
        self.assertEqual(plan.env["HELICOPTER_LIGHTEEVAL_ASSERT_LOCAL_SOURCE"], "1")

    def test_g1h_plan_uses_doc_generation_size_and_aliases_tasks(self) -> None:
        loaded_config = load_example_config()
        loaded_config["lighteval"]["g1h"] = {
            "metric": "avg",
            "prompt_style": "naive",
            "zero_shot": True,
            "avg_k": 8,
            "rollout_n": 8,
            "generation_size": 4096,
            "gpass_k": 16,
            "gpass_n": 48,
            "gpass_generation_size": 8192,
        }
        plan = commands.build_lighteval_plan(
            lighteval_args(tasks="aime24,aime24_avg,math_500"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[6], "g1h__aime24_avg|0,g1h__math_500|0")
        self.assertNotIn("max_new_tokens", plan.command[5])
        sampling_json = plan.env.get("HELICOPTER_VLLM_SAMPLING_JSON")
        if sampling_json:
            self.assertNotIn("max_tokens", json.loads(sampling_json))
        policy = json.loads(plan.env["HELICOPTER_LIGHTEEVAL_G1H_POLICY"])
        self.assertEqual(policy["selected_tasks"], ["aime24_avg", "math_500"])

    @unittest.skip("aggregate preset configs were replaced by standalone benchmark TOMLs")
    def test_server_preset_passes_prompt_config_and_toml_k_to_child(self) -> None:
        loaded_config, _ = config.load_config(ROOT, "configs/presets/naive-cot.toml")
        plan = commands.build_lighteval_plan(
            lighteval_args(
                model="deployed",
                tasks="gsm8k|0,arc:challenge|0",
                config="configs/presets/naive-cot.toml",
                base_url="http://127.0.0.1:29573/v1",
                api_key="test-key",
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        policy = json.loads(plan.env["HELICOPTER_LIGHTEEVAL_G1H_POLICY"])
        self.assertEqual(policy["avg_k"], loaded_config["lighteval"]["g1h"]["avg_k"])
        self.assertEqual(policy["rollout_n"], loaded_config["lighteval"]["g1h"]["rollout_n"])
        self.assertEqual(policy["target_generations_per_benchmark"], 5000)
        self.assertEqual(policy["large_benchmark_generation_threshold"], 20000)
        self.assertEqual(policy["large_benchmark_sample_rate"], 0.2)
        self.assertEqual(plan.env["HELICOPTER_PROMPT_TEMPLATE"], "User: {query}\nAssistant: <think")
        request_policy = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])
        self.assertEqual(
            request_policy["tasks"]["gsm8k"]["prompt_template"],
            "User: {query}\nAssistant: <think",
        )
        self.assertEqual(
            request_policy["tasks"]["arc:challenge"]["prompt_template"],
            "User: {query}\nAssistant: <think",
        )
        self.assertEqual(request_policy["tasks"]["gsm8k"]["stop"], ["\nUser:"])
        self.assertEqual(request_policy["tasks"]["arc:challenge"]["stop"], ["\nUser:"])
        self.assertEqual(plan.env["HELICOPTER_SCOREBOARD_PROMPT_MODE"], "naive_cot")
        self.assertEqual(
            plan.env["HELICOPTER_SCOREBOARD_CONFIG_PATH"],
            str((ROOT / "configs/presets/naive-cot.toml").resolve()),
        )

    @unittest.skip("aggregate preset configs were replaced by standalone benchmark TOMLs")
    def test_cot_server_preset_rejects_coding_and_instruction_domains(self) -> None:
        loaded_config, _ = config.load_config(ROOT, "configs/presets/naive-cot.toml")
        for task in ("lcb:codegeneration_v6|0", "ifeval|0"):
            with self.subTest(task=task), self.assertRaisesRegex(
                SystemExit, "prompt mode does not allow"
            ):
                commands.build_lighteval_plan(
                    lighteval_args(
                        model="deployed",
                        tasks=task,
                        config="configs/presets/naive-cot.toml",
                    ),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    @unittest.skip("aggregate preset configs were replaced by standalone benchmark TOMLs")
    def test_nocot_preset_distinguishes_code_generation_from_cs_choices(self) -> None:
        for preset, expected_prefix in (
            ("configs/presets/naive-nocot.toml", "User:"),
            ("configs/presets/normal-nocot.toml", "User✿"),
        ):
            with self.subTest(preset=preset):
                loaded_config, _ = config.load_config(ROOT, preset)
                plan = commands.build_lighteval_plan(
                    lighteval_args(
                        model="deployed",
                        tasks="lcb:codegeneration|0,mmlu:machine_learning|0,ceval_zho_mcf:college_programming|0",
                        config=preset,
                    ),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

                tasks = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]
                self.assertEqual(tasks["lcb:codegeneration"]["format"], "python_program")
                self.assertTrue(tasks["lcb:codegeneration"]["prompt_template"].endswith("```python"))
                self.assertEqual(tasks["lcb:codegeneration"]["stop"][-1], "\n```")
                self.assertTrue(tasks["lcb:codegeneration"]["prompt_template"].startswith(expected_prefix))
                for task_name in ("mmlu:machine_learning", "ceval_zho_mcf:college_programming"):
                    self.assertEqual(tasks[task_name]["format"], "choice")
                    self.assertNotIn("```", tasks[task_name]["prompt_template"])
                    self.assertNotIn("option letter", tasks[task_name]["prompt_template"])
                    self.assertTrue(tasks[task_name]["prompt_template"].startswith(expected_prefix))
                    self.assertEqual(tasks[task_name]["sampling"]["max_tokens"], 128)

    @unittest.skip("aggregate preset configs were replaced by standalone benchmark TOMLs")
    def test_nocot_request_format_recognition_is_toml_driven(self) -> None:
        preset = "configs/presets/naive-nocot.toml"
        loaded_config, _ = config.load_config(ROOT, preset)
        code_format = loaded_config["prompt"]["formats"]["python_program"]
        code_format["tasks"] = ["mmlu:machine_learning"]
        code_format["task_prefixes"] = []
        plan = commands.build_lighteval_plan(
            lighteval_args(
                model="deployed",
                tasks="lcb:codegeneration|0,mmlu:machine_learning|0",
                config=preset,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        tasks = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]
        # famous120 owns this selection in its per-benchmark TOML, so mutating
        # the shared preset's task list cannot silently change the benchmark
        # contract.
        self.assertEqual(tasks["mmlu:machine_learning"]["format"], "choice")
        self.assertEqual(tasks["lcb:codegeneration"]["format"], "python_program")

    @unittest.skip("aggregate preset configs were replaced by standalone benchmark TOMLs")
    def test_nocot_server_preset_keeps_domain_token_budgets_in_toml(self) -> None:
        loaded_config, _ = config.load_config(ROOT, "configs/presets/normal-nocot.toml")
        plan = commands.build_lighteval_plan(
            lighteval_args(
                model="deployed",
                tasks="gsm8k|0,lcb:codegeneration_v6|0,ifeval|0,arc:challenge|0",
                config="configs/presets/normal-nocot.toml",
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        tasks = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]
        self.assertEqual(tasks["gsm8k"]["sampling"]["max_tokens"], 1024)
        self.assertEqual(tasks["lcb:codegeneration_v6"]["sampling"]["max_tokens"], 4096)
        self.assertEqual(tasks["ifeval"]["sampling"]["max_tokens"], 1024)
        self.assertEqual(tasks["arc:challenge"]["sampling"]["max_tokens"], 128)

    def test_standalone_config_owns_prompt_and_sampling_controls(self) -> None:
        config_path = "configs/benchmarks/g1h/math/050_gsm8k.toml"
        loaded_config, _ = config.load_config(ROOT, config_path)
        config.merge_model_catalog(
            loaded_config,
            root=ROOT,
            catalog_path="configs/models/g1h-dual-replica.toml",
        )
        loaded_config["prompt"]["mode"] = "naive_cot"
        loaded_config["prompt"]["template"] = "User: {query}\nAssistant: <think"
        plan = commands.build_lighteval_plan(
            lighteval_args(
                model="deployed",
                tasks="gsm8k|0",
                config=config_path,
                base_url="http://127.0.0.1:29573/v1",
                api_key="test-key",
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )
        policy = json.loads(plan.env["HELICOPTER_LIGHTEEVAL_G1H_POLICY"])
        evaluation = loaded_config["_benchmark_specs"]["gsm8k"]["evaluation"]
        self.assertEqual(policy["avg_k"], evaluation["avg_k"])
        self.assertEqual(policy["rollout_n"], evaluation["rollout_n"])
        request = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"]["gsm8k"]
        self.assertEqual(request["prompt_template"], "User: {query}\n\nAssistant: <think")
        self.assertEqual(request["sampling"]["context_budget"], 10240)
        self.assertEqual(plan.env["HELICOPTER_SCOREBOARD_PROMPT_MODE"], "naive_cot")

    def test_prompt_mode_override_isolated_and_uses_official_stops(self) -> None:
        config_path = "configs/benchmarks/g1h/math/058_math_500.toml"
        loaded_config, _ = config.load_config(ROOT, config_path)
        config.merge_model_catalog(
            loaded_config,
            root=ROOT,
            catalog_path="configs/models/g1h-dual-replica.toml",
        )
        naive_plan = commands.build_lighteval_plan(
            lighteval_args(
                model="g1h-2.9b",
                tasks="math_500|0",
                config=config_path,
                prompt_mode="naive_nocot",
                base_url="http://127.0.0.1:19329/v1",
                api_key="rwkv-skills",
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )
        naive_request = json.loads(
            naive_plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"]
        )["tasks"]["math_500"]
        self.assertEqual(naive_request["prompt_mode"], "naive_nocot")
        self.assertEqual(naive_request["prompt_template"], "User: {query}\n\nAssistant: <think></think")
        self.assertEqual(naive_request["stop"], ["\nUser:"])
        self.assertEqual(naive_plan.env["HELICOPTER_PROMPT_MODE"], "naive_nocot")

        normal_plan = commands.build_lighteval_plan(
            lighteval_args(
                model="g1h-2.9b",
                tasks="math_500|0",
                config=config_path,
                prompt_mode="normal_nocot",
                base_url="http://127.0.0.1:19329/v1",
                api_key="rwkv-skills",
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )
        normal_request = json.loads(
            normal_plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"]
        )["tasks"]["math_500"]
        self.assertEqual(normal_request["prompt_mode"], "normal_nocot")
        self.assertTrue(normal_request["prompt_template"].startswith("User✿"))
        self.assertIn("\nUser:", normal_request["stop"])
        self.assertIn("✿", normal_request["stop"])

    def test_benchmark_mode_templates_decode_newline_as_0x0a(self) -> None:
        config_path = "configs/benchmarks/g1h/knowledge/073_mmlu_miscellaneous.toml"
        loaded_config, _ = config.load_config(ROOT, config_path)
        template = commands.prompt_template_for_mode(loaded_config["prompt"])

        self.assertEqual(template, "User✿{query}✿\nBot✿<think></think")
        self.assertIn("\n", template)
        self.assertNotIn("\\n", template)

    def test_standalone_file_owns_request_format(self) -> None:
        cases = (
            ("configs/benchmarks/g1h/coding/091_lcb_codegeneration_v3.toml", "python_program"),
            ("configs/benchmarks/g1h/knowledge/052_mmlu_business_ethics.toml", "choice"),
        )
        for config_path, expected_format in cases:
            with self.subTest(config_path=config_path):
                loaded_config, _ = config.load_config(ROOT, config_path)
                config.merge_model_catalog(
                    loaded_config,
                    root=ROOT,
                    catalog_path="configs/models/g1h-dual-replica.toml",
                )
                task = next(iter(loaded_config["_benchmark_specs"]))
                plan = commands.build_lighteval_plan(
                    lighteval_args(model="deployed", tasks=f"{task}|0", config=config_path),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )
                request = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])["tasks"][task]
                self.assertEqual(request["format"], expected_format)
                if expected_format == "python_program":
                    self.assertTrue(request["prompt_template"].endswith("```python"))
                else:
                    self.assertNotIn("```", request["prompt_template"])

    def test_avg_controls_remain_in_standalone_benchmark_toml(self) -> None:
        config_path = "configs/benchmarks/g1h/knowledge/046_gpqa_diamond.toml"
        loaded_config, _ = config.load_config(ROOT, config_path)
        spec = loaded_config["_benchmark_specs"]["gpqa:diamond"]
        self.assertNotIn("g1h", loaded_config.get("lighteval", {}))
        self.assertEqual(spec["evaluation"]["metric"], "avg")
        self.assertEqual(spec["evaluation"]["pass_k"], 1)
        self.assertEqual(spec["evaluation"]["avg_k"], spec["evaluation"]["rollout_n"])
        self.assertEqual(spec["sampling"]["context_budget"], 10240)

    def test_lighteval_plan_cli_max_new_tokens_overrides_config(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_plan(
            lighteval_args(max_new_tokens=128),
            root=ROOT,
            env={"HELICOPTER_EVAL_MAX_NEW_TOKENS": "256"},
            config=loaded_config,
        )

        self.assertIn("generation_parameters={max_new_tokens:128}", plan.command[5])

    def test_lighteval_plan_forwards_canonical_vllm_sampling(self) -> None:
        loaded_config = load_example_config()
        loaded_config["lighteval"] = {
            key: value
            for key, value in loaded_config["lighteval"].items()
            if key != "max_new_tokens"
        }
        loaded_config["sampling"] = {
            "max_tokens": 512,
            "temperature": 0.8,
            "top_p": 0.35,
            "top_k": 40,
            "presence_penalty": 0.65,
            "frequency_penalty": 0.25,
            "penalty_decay": 0.99,
        }

        plan = commands.build_lighteval_plan(
            lighteval_args(),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        model_args = plan.command[5]
        self.assertIn("use_cache=false", model_args)
        self.assertIn("generation_parameters={", model_args)
        self.assertIn("max_new_tokens:512", model_args)
        self.assertIn("temperature:0.8", model_args)
        self.assertIn("top_p:0.35", model_args)
        self.assertIn("top_k:40", model_args)
        self.assertNotIn("penalty_decay", model_args)
        self.assertEqual(
            json.loads(plan.env["HELICOPTER_VLLM_SAMPLING_JSON"]),
            loaded_config["sampling"],
        )

    def test_lighteval_plan_keeps_api_key_out_of_command(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_plan(
            lighteval_args(base_url="https://example.test/v1", api_key="secret-token"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertNotIn("secret-token", " ".join(plan.command))
        self.assertEqual(plan.env["OPENAI_API_KEY"], "secret-token")

    def test_lighteval_performance_metrics_url_uses_endpoint_root(self) -> None:
        self.assertEqual(
            performance.derive_metrics_url("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            performance.derive_metrics_url("http://127.0.0.1:8000/custom/v1"),
            "http://127.0.0.1:8000/custom/metrics",
        )

    def test_lighteval_performance_reads_equals_output_dir_option(self) -> None:
        self.assertEqual(
            performance.output_dir_from_command(["lighteval", "--output-dir=results/run"]),
            Path("results/run"),
        )

    def test_lighteval_performance_report_derives_run_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result_dir = output_dir / "results/openai/model"
            result_dir.mkdir(parents=True)
            result_file = result_dir / "results_2026-07-03T00-00-00.json"
            result_file.write_text(
                json.dumps(
                    {
                        "config_general": {
                            "model_name": "openai/g1g-7.2b",
                            "max_samples": 2,
                            "total_evaluation_time_secondes": "10.0",
                        },
                        "results": {
                            "gsm8k|0": {"acc": 0.5},
                            "all": {"acc": 0.5},
                        },
                    }
                )
            )

            report = performance.build_performance_report(
                command=["python", "-m", "lighteval"],
                exit_code=0,
                output_dir=output_dir,
                started_at_epoch=1.0,
                ended_at_epoch=11.0,
                metrics_url="http://127.0.0.1:8000/metrics",
                metrics_before={
                    "vllm:prompt_tokens_total": 100,
                    "vllm:generation_tokens_total": 10,
                },
                metrics_after={
                    "vllm:prompt_tokens_total": 130,
                    "vllm:generation_tokens_total": 30,
                },
            )
            self.assertEqual(performance.extract_lighteval_score_metrics(report["source_files"]["results"]), {"acc": 0.5})

        self.assertEqual(report["samples_completed"], 2)
        self.assertEqual(report["jobs_completed"], 1)
        self.assertEqual(report["models_completed"], 1)
        self.assertEqual(report["benchmarks_completed"], 1)
        self.assertEqual(report["prompt_tokens"], 30)
        self.assertEqual(report["generation_tokens"], 20)
        self.assertEqual(report["total_tokens"], 50)
        self.assertEqual(report["samples_per_hour"], 720.0)
        self.assertEqual(report["jobs_per_hour"], 360.0)
        self.assertEqual(report["models_per_day"], 8640.0)
        self.assertEqual(report["benchmarks_per_day"], 8640.0)
        self.assertEqual(report["tokens_per_second"], 5.0)
        self.assertEqual(report["completed_runs/day"], 8640.0)

    def test_lighteval_performance_metrics_embeds_scoreboard_subset(self) -> None:
        report = {
            "elapsed_seconds": 5.0,
            "samples_completed": 1,
            "jobs_completed": 1,
            "models_completed": 1,
            "benchmarks_completed": 1,
            "completed_runs": 1,
            "tokens_per_second": None,
            "samples_per_hour": 720.0,
            "jobs_per_hour": 720.0,
            "models_per_day": 17280.0,
            "benchmarks_per_day": 17280.0,
            "completed_runs/day": 17280.0,
        }

        embedded = performance.performance_metrics_from_report(report)

        self.assertEqual(embedded["samples_per_hour"], 720.0)
        self.assertEqual(embedded["tokens_per_second"], None)
        self.assertNotIn("command", embedded)

    def test_completions_performance_report_derives_profile_rates(self) -> None:
        results = [
            performance.CompletionRequestResult(
                True,
                0.10,
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
            ),
            performance.CompletionRequestResult(
                True,
                0.30,
                prompt_tokens=120,
                completion_tokens=20,
                total_tokens=140,
            ),
            performance.CompletionRequestResult(
                False,
                0.20,
                error="TimeoutError: slow",
            ),
        ]

        report = performance.build_completions_performance_report(
            model_name="g1d-0.4b",
            base_url="http://127.0.0.1:8000/v1",
            profile="decode",
            prompt_tokens_target=128,
            output_tokens=256,
            requests=3,
            concurrency=2,
            request_rate=None,
            timeout_s=120.0,
            ignore_eos=True,
            started_at_epoch=1.0,
            ended_at_epoch=3.0,
            results=results,
        )

        self.assertEqual(report["kind"], "openai_completions")
        self.assertEqual(report["successful_requests"], 2)
        self.assertEqual(report["failed_requests"], 1)
        self.assertEqual(report["error_rate"], 1 / 3)
        self.assertEqual(report["prompt_tokens"], 220)
        self.assertEqual(report["completion_tokens"], 30)
        self.assertEqual(report["total_tokens"], 250)
        self.assertEqual(report["request_throughput"], 1.0)
        self.assertEqual(report["completion_tokens_per_second"], 15.0)
        self.assertEqual(report["e2e_latency_seconds"]["p50"], 0.10)
        self.assertEqual(report["errors"], {"TimeoutError: slow": 1})

    def test_completions_performance_dry_run_uses_profile_defaults(self) -> None:
        loaded = load_example_config()
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = performance.run_completions_performance(
                perf_args(profile="prefill", output="tmp/perf.json"),
                root=ROOT,
                env={},
                config=loaded,
            )

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("profile=prefill", output)
        self.assertIn("prompt_tokens=2048", output)
        self.assertIn("output_tokens=8", output)
        self.assertIn(str(ROOT / "tmp/perf.json"), output)

    def test_lighteval_tasks_plan_lists_registry(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(load_tasks_multilingual=True),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:5], ["-m", "lighteval", "tasks", "list"])
        self.assertIn("--load-tasks-multilingual", plan.command)
        self.assertEqual(
            command_options(plan.command)["--custom-tasks"],
            str(ROOT / "src/cli/helicopter_cli/lighteval_policy_tasks.py"),
        )

    def test_lighteval_tasks_show_config_uses_local_compat_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(task_action="inspect", tasks="gsm8k", show_config=True, num_samples=1),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "inspect"])
        self.assertIn("--show-config", plan.command)
        self.assertEqual(command_options(plan.command)["--num-samples"], "1")

    def test_lighteval_tasks_export_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="export",
                load_tasks_multilingual=True,
                output="tmp/tasks.txt",
                contains=["gsm"],
                limit=2,
                include_supersets=True,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "export"])
        self.assertIn("--load-multilingual", plan.command)
        self.assertIn("--include-supersets", plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--output"], "tmp/tasks.txt")
        self.assertEqual(options["--contains"], "gsm")
        self.assertEqual(options["--limit"], "2")

    def test_lighteval_tasks_export_filters_registry_rows(self) -> None:
        class FakeRegistry:
            _task_registry = {"gsm8k": object(), "mmlu": object(), "tiny:gsm8k": object()}
            _task_superset_dict = {"tiny": ("tiny:gsm8k",)}

        with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
            rows = lighteval_tasks.selected_task_rows(
                Namespace(
                    custom_tasks=None,
                    load_multilingual=False,
                    contains=["gsm"],
                    limit=None,
                    include_supersets=True,
                )
            )

        self.assertEqual(rows, [("task", "gsm8k"), ("task", "tiny:gsm8k")])
        self.assertEqual(
            lighteval_tasks.format_export(rows, "jsonl"),
            '{"kind": "task", "task": "gsm8k"}\n{"kind": "task", "task": "tiny:gsm8k"}\n',
        )

    def test_lighteval_tasks_coverage_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="coverage",
                source="benchmarks.txt",
                source_format="text",
                output="tmp/coverage.jsonl",
                format="jsonl",
                candidate_limit=7,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "coverage"])
        options = command_options(plan.command)
        self.assertEqual(options["--source"], str(ROOT / "benchmarks.txt"))
        self.assertEqual(options["--source-format"], "text")
        self.assertEqual(options["--output"], "tmp/coverage.jsonl")
        self.assertEqual(options["--format"], "jsonl")
        self.assertEqual(options["--candidate-limit"], "7")

    def test_lighteval_tasks_judges_uses_local_registry_wrapper(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_tasks_plan(
            lighteval_tasks_args(
                task_action="judges",
                tasks="aime24",
                load_tasks_multilingual=True,
                output="tmp/judges.jsonl",
                format="jsonl",
                contains=["aime"],
                limit=1,
            ),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:4], ["-m", "helicopter_cli.lighteval_tasks", "judges"])
        self.assertIn("--load-multilingual", plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--custom-tasks"], str(ROOT / "src/cli/helicopter_cli/lighteval_policy_tasks.py"))
        self.assertEqual(options["--output"], "tmp/judges.jsonl")
        self.assertEqual(options["--format"], "jsonl")
        self.assertEqual(options["--contains"], "aime")
        self.assertEqual(options["--limit"], "1")
        self.assertIn("aime24", plan.command)

    def test_lighteval_tasks_coverage_rejects_filter_flags(self) -> None:
        loaded_config = load_example_config()
        for overrides in (
            {"contains": ["gsm"]},
            {"limit": 5},
            {"include_supersets": True},
        ):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="coverage", source="benchmarks.txt", **overrides),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_lighteval_tasks_export_rejects_unsupported_format(self) -> None:
        loaded_config = load_example_config()
        for fmt in ("summary", "tasks"):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="export", format=fmt),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_lighteval_tasks_judges_rejects_tasks_format(self) -> None:
        loaded_config = load_example_config()
        with self.assertRaises(SystemExit):
            commands.build_lighteval_tasks_plan(
                lighteval_tasks_args(task_action="judges", tasks="aime24", format="tasks"),
                root=ROOT,
                env={},
                config=loaded_config,
            )

    def test_lighteval_tasks_judges_rejects_include_supersets(self) -> None:
        loaded_config = load_example_config()
        with self.assertRaises(SystemExit):
            commands.build_lighteval_tasks_plan(
                lighteval_tasks_args(task_action="judges", tasks="aime24", include_supersets=True),
                root=ROOT,
                env={},
                config=loaded_config,
            )

    def test_lighteval_tasks_list_rejects_filter_and_output_flags(self) -> None:
        loaded_config = load_example_config()
        for overrides in ({"output": "tmp/tasks.txt"}, {"contains": ["gsm"]}, {"limit": 3}):
            with self.assertRaises(SystemExit):
                commands.build_lighteval_tasks_plan(
                    lighteval_tasks_args(task_action="list", **overrides),
                    root=ROOT,
                    env={},
                    config=loaded_config,
                )

    def test_normalize_openai_base_url_appends_v1_to_bare_host(self) -> None:
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000"),
            "http://host:8000/v1",
        )
        # Existing path (or /v1) is preserved, only trailing slashes trimmed.
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000/v1/"),
            "http://host:8000/v1",
        )
        self.assertEqual(
            commands.normalize_openai_base_url("http://host:8000/custom"),
            "http://host:8000/custom",
        )

    def test_lighteval_tasks_coverage_resolves_registry_rows(self) -> None:
        mmmlu_targets = lighteval_tasks.OFFICIAL_LIGHTEVAL_ALIASES["mmmlu"]

        class FakeRegistry:
            _task_registry = {
                "gpqa:diamond": object(),
                "gsm8k": object(),
                "ifbench_multiturn": object(),
                "ifbench_test": object(),
                "supergpqa": object(),
                "tiny:gsm8k": object(),
            }
            _task_superset_dict = {
                "lcb": ("lcb:codegeneration",),
                "mmlu": ("mmlu:abstract_algebra",),
                **{target: (f"{target}:abstract_algebra",) for target in mmmlu_targets},
            }

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "benchmarks.txt"
            source.write_text(
                "gsm8k,maths\n"
                "gpqa_diamond,knowledge\n"
                "mmlu,knowledge\n"
                "mmmlu,knowledge\n"
                "supergpqa,knowledge\n"
                "ifbench,instruction_following\n"
                "livecodebench,coding\n"
                "missing_one,maths\n"
            )
            with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
                rows = lighteval_tasks.coverage_rows(
                    Namespace(
                        custom_tasks=None,
                        load_multilingual=False,
                        source=str(source),
                        source_format="text",
                        candidate_limit=3,
                    )
                )

        self.assertEqual(rows[0].status, "exact_task")
        self.assertEqual(rows[1].status, "normalized_task")
        self.assertEqual(rows[1].targets, ("gpqa:diamond",))
        self.assertEqual(rows[2].status, "exact_superset")
        self.assertEqual(rows[3].status, "alias_superset_list")
        self.assertEqual(rows[3].targets, mmmlu_targets)
        self.assertEqual(rows[4].status, "exact_task")
        self.assertEqual(rows[5].status, "alias_task_list")
        self.assertEqual(rows[5].targets, ("ifbench_test", "ifbench_multiturn"))
        self.assertEqual(rows[6].status, "alias_superset")
        self.assertEqual(rows[6].targets, ("lcb",))
        self.assertEqual(rows[7].status, "missing")
        self.assertIn("direct\t7\n", lighteval_tasks.format_coverage(rows, "summary"))
        self.assertIn("not_direct\t1\n", lighteval_tasks.format_coverage(rows, "summary"))
        self.assertEqual(
            lighteval_tasks.format_coverage(rows, "tasks"),
            "gsm8k\n"
            "gpqa:diamond\n"
            "mmlu\n"
            + "".join(f"{target}\n" for target in mmmlu_targets)
            + "supergpqa\n"
            "ifbench_test\n"
            "ifbench_multiturn\n"
            "lcb\n",
        )

    def test_lighteval_tasks_json_source_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "agent_benchmarks.json"
            source.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "name": "swe_bench_verified",
                                "field": "agent",
                                "display_name": "SWE-bench Verified",
                                "run_status": "local_lighteval_proxy_available",
                                "priority": "primary",
                            }
                        ],
                        "excluded": [
                            {"name": "gpqa_diamond", "field": "knowledge"},
                        ],
                    }
                )
            )

            rows = lighteval_tasks.load_source_benchmarks(str(source), "auto")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "swe_bench_verified")
        self.assertEqual(rows[0].field, "agent")
        self.assertEqual(rows[0].metadata["run_status"], "local_lighteval_proxy_available")
        self.assertEqual(rows[0].metadata["priority"], "primary")

        coverage = [
            lighteval_tasks.CoverageRow(
                source=rows[0].name,
                field=rows[0].field,
                status="missing",
                target_kind=None,
                targets=(),
                candidates=(),
                metadata=rows[0].metadata,
            )
        ]
        json_row = json.loads(lighteval_tasks.format_coverage(coverage, "jsonl"))
        self.assertEqual(json_row["metadata"]["display_name"], "SWE-bench Verified")
        self.assertIn("local_lighteval_proxy_available", lighteval_tasks.format_coverage(coverage, "text"))

    def test_agent_benchmark_source_excludes_non_agent_benchmarks(self) -> None:
        raw_source = json.loads((ROOT / "benchmarks/agent_benchmarks.json").read_text())
        rows = lighteval_tasks.load_source_benchmarks(
            str(ROOT / "benchmarks/agent_benchmarks.json"),
            "auto",
        )
        names = {row.name for row in rows}
        pipelines = {
            pipeline["name"]: set(pipeline["benchmarks"])
            for pipeline in raw_source["pipelines"]
        }
        row_pipeline_names = {row.metadata["pipeline"] for row in rows}
        benchmark_names = {benchmark["name"] for benchmark in raw_source["benchmarks"]}
        excluded_names = {benchmark["name"] for benchmark in raw_source["excluded"]}

        self.assertIn("swe_bench_verified", names)
        self.assertIn("terminal_bench_2_1", names)
        self.assertIn("mcp_atlas", names)
        self.assertIn("hle_with_tools", names)
        self.assertIn("hy_euler_pro", names)
        self.assertNotIn("gpqa_diamond", names)
        self.assertNotIn("hle_no_tools", names)
        self.assertNotIn("cl_bench", names)
        self.assertNotIn("hy_math", names)
        self.assertTrue(all(row.field == "agent" for row in rows))
        self.assertEqual(
            set(pipelines),
            {
                "coding_agent",
                "search_agent",
                "tool_mcp_agent",
                "office_enterprise_workflow_agent",
                "stem_tool_agent",
            },
        )
        self.assertTrue(all(row.metadata["pipeline"] in pipelines for row in rows))
        self.assertEqual(row_pipeline_names, set(pipelines))
        self.assertEqual(set().union(*pipelines.values()), benchmark_names)
        self.assertTrue(benchmark_names.isdisjoint(excluded_names))
        self.assertEqual(
            pipelines["coding_agent"],
            {
                "swe_bench_verified",
                "swe_bench_multilingual",
                "swe_bench_pro",
                "terminal_bench_2_1",
                "nl2repo",
                "deepswe",
                "hy_backend_2_0",
                "hy_swe_max",
            },
        )
        self.assertEqual(pipelines["search_agent"], {"browsecomp", "wide_search", "deepsearchqa"})
        self.assertEqual(
            pipelines["tool_mcp_agent"],
            {"mcp_atlas", "toolathlon", "skillsbench", "hy_skillsworld"},
        )
        self.assertEqual(
            pipelines["office_enterprise_workflow_agent"],
            {
                "apex_agents",
                "claweval",
                "wildclawbench",
                "hy_companybench",
                "prodbench",
                "hy_finmodelbench",
                "e_bench",
            },
        )
        self.assertEqual(pipelines["stem_tool_agent"], {"hle_with_tools", "hy_euler_pro"})

    def test_non_fc_lighteval_catalog_builder_has_100_diverse_recognized_tasks_per_domain(self) -> None:
        from helicopter_cli.non_fc_lighteval_catalog import build_manifest

        raw_source = build_manifest(root=ROOT)
        rows = raw_source["benchmarks"]
        names = {row["name"] for row in rows}
        counts = Counter(row["field"] for row in rows)
        disallowed = (
            "bfcl",
            "api_bank",
            "complexfuncbench",
            "toolalpaca",
            "function_call",
            "tool_call",
            "mcp_bench",
            "tau_bench",
            "agentbench",
            "browsecomp",
            "swe_bench",
        )

        self.assertEqual(raw_source["target_per_domain"], 100)
        self.assertEqual(raw_source["scope"], "direct_hf_lighteval_non_function_calling")
        self.assertEqual(raw_source["scoreboard_domain_scope"]["included"], ["math", "coding", "instruction_following", "knowledge"])
        self.assertEqual({row["field"] for row in raw_source["scoreboard_domain_scope"]["excluded"]}, {"agent", "function_call"})
        self.assertEqual(
            counts,
            Counter({"math": 100, "coding": 100, "instruction_following": 100, "knowledge": 100}),
        )
        self.assertEqual(len(names), len(rows))
        self.assertNotIn("agent", counts)
        self.assertNotIn("function_call", counts)
        self.assertIn("natural_questions", names)
        self.assertIn("squad_v2", names)
        self.assertIn("truthfulqa:mc", names)
        self.assertNotIn("mathqa", names)
        self.assertIn("gpqa:diamond", names)
        self.assertNotIn("ifeval-fr", names)
        self.assertNotIn("qasper", names)
        self.assertNotIn("the_pile:arxiv", names)
        self.assertTrue(all(row["source_family"] for row in rows))
        for field in ("math", "coding", "instruction_following", "knowledge"):
            families = {row["source_family"] for row in rows if row["field"] == field}
            self.assertGreaterEqual(len(families), 4)
        self.assertFalse(any(any(token in row["name"].lower() for token in disallowed) for row in rows))

    def test_agent_harness_source_binds_every_benchmark_to_profile(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        names = {row.name for row in source.benchmarks}

        self.assertIn("swe_bench_verified", names)
        self.assertIn("terminal_bench_2_1", names)
        self.assertIn("mcp_atlas", names)
        self.assertIn("hle_with_tools", names)
        self.assertIn("swebench_official_docker", source.profiles)
        self.assertIn("terminal_bench_official_docker", source.profiles)
        self.assertTrue(all(row.harness_profile in source.profiles for row in source.benchmarks))
        self.assertEqual(
            source.profiles["swebench_official_docker"].entrypoint,
            "python -m swebench.harness.run_evaluation",
        )
        self.assertEqual(source.profiles["terminal_bench_official_docker"].sandbox, "docker")

    def test_agent_harness_preflight_reports_missing_required_tool(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)

        with mock.patch.object(agent_harness.shutil, "which", return_value=None):
            rows = agent_harness.preflight_rows(
                source,
                pipeline=None,
                benchmark="swe_bench_verified",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "blocked")
        self.assertEqual(rows[0].missing_tools, ("docker",))

    def test_agent_harness_preflight_distinguishes_local_proxy_from_official_harness(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)

        with mock.patch.object(agent_harness.shutil, "which", return_value="/usr/bin/docker"):
            rows = agent_harness.preflight_rows(
                source,
                pipeline=None,
                benchmark="swe_bench_verified",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].status, "local_proxy_available_official_harness_required")

    def test_agent_harness_swebench_plan_uses_official_docker_harness(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        args = Namespace(
            benchmark="swe_bench_verified",
            model="g1d-0.4b",
            base_url="http://127.0.0.1:8000/v1",
            output_dir="tmp/agent",
            n_concurrent=2,
            run_id="unit",
        )

        plan = agent_harness.plan_for_benchmark(
            source,
            root=ROOT,
            env={},
            config=load_example_config(),
            args=args,
        )

        self.assertEqual(plan["benchmark"]["name"], "swe_bench_verified")
        self.assertEqual(plan["harness_profile"]["kind"], "swebench")
        self.assertEqual(plan["steps"][0]["schema"]["model_patch"], "unified diff patch generated by the coding agent")
        self.assertEqual(
            plan["steps"][1]["command"][:3],
            ["python", "-m", "swebench.harness.run_evaluation"],
        )
        self.assertIn("SWE-bench/SWE-bench_Verified", plan["steps"][1]["command"])
        self.assertIn("--predictions_path", plan["steps"][1]["command"])
        self.assertIn("--max_workers", plan["steps"][1]["command"])
        self.assertIn("2", plan["steps"][1]["command"])

    def test_agent_harness_run_writes_plan_for_external_harness_without_fake_score(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                benchmark="deepswe",
                pipeline=None,
                format="jsonl",
                model="g1d-0.4b",
                base_url="http://127.0.0.1:8000/v1",
                output_dir=tmp,
                n_concurrent=1,
                run_id=None,
                max_samples=None,
                no_server=True,
                keep_server=False,
                server_timeout=600,
                allow_proxy=False,
                dry_run=False,
            )

            rows, exit_code = agent_harness.run_agent_benchmarks(
                source,
                root=ROOT,
                env={},
                config=load_example_config(),
                args=args,
            )
            plan_path = Path(rows[0]["plan_path"])

            self.assertEqual(exit_code, 1)
            self.assertEqual(rows[0]["status"], "external_harness_not_implemented")
            self.assertTrue(plan_path.exists())
            self.assertEqual(json.loads(plan_path.read_text(encoding="utf-8"))["benchmark"]["name"], "deepswe")

    def test_agent_harness_run_blocks_browsecomp_proxy_unless_allowed(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                benchmark="browsecomp",
                pipeline=None,
                format="jsonl",
                model="g1d-0.4b",
                base_url="http://127.0.0.1:8000/v1",
                output_dir=tmp,
                n_concurrent=1,
                run_id=None,
                max_samples=1,
                no_server=True,
                keep_server=False,
                server_timeout=600,
                allow_proxy=False,
                dry_run=False,
            )

            rows, exit_code = agent_harness.run_agent_benchmarks(
                source,
                root=ROOT,
                env={},
                config=load_example_config(),
                args=args,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(rows[0]["status"], "blocked_proxy")
        self.assertIn("not a browser-runtime score", rows[0]["message"])

    def test_agent_harness_run_browsecomp_proxy_dry_run_emits_lighteval_command(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        args = Namespace(
            benchmark="browsecomp",
            pipeline=None,
            format="jsonl",
            model="g1d-0.4b",
            base_url="http://127.0.0.1:8000/v1",
            output_dir="tmp/agent",
            n_concurrent=1,
            run_id=None,
            max_samples=1,
            no_server=True,
            keep_server=False,
            server_timeout=600,
            allow_proxy=True,
            dry_run=True,
        )

        rows, exit_code = agent_harness.run_agent_benchmarks(
            source,
            root=ROOT,
            env={},
            config=load_example_config(),
            args=args,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(rows[0]["status"], "dry_run")
        self.assertEqual(rows[0]["command"][:5], ["helicopter", "eval", "run", "g1d-0.4b", "browsecomp"])
        self.assertIn("--no-server", rows[0]["command"])

    def test_agent_format_extracts_swebench_predictions_from_rwkv_response(self) -> None:
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        records = [
            {
                "sample_id": "repo__project-1",
                "response_message": {
                    "role": "assistant",
                    "content": f"Here is the fix:\n```diff\n{patch}```",
                    "finish_reason": "stop",
                },
            }
        ]

        rows, errors = agent_format.swebench_prediction_rows(
            records,
            model="g1d-0.4b",
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            rows,
            [
                {
                    "instance_id": "repo__project-1",
                    "model_name_or_path": "g1d-0.4b",
                    "model_patch": patch,
                }
            ],
        )

    def test_agent_format_reports_missing_swebench_patch(self) -> None:
        rows, errors = agent_format.swebench_prediction_rows(
            [{"instance_id": "repo__project-1", "output": "I cannot fix this."}],
            model="g1d-0.4b",
        )

        self.assertEqual(rows, [])
        self.assertEqual(errors[0].reason, "missing unified diff patch")
        self.assertIn("repo__project-1", agent_format.conversion_errors_text(errors))

    def test_agent_harness_convert_writes_swebench_predictions_jsonl(self) -> None:
        source = agent_harness.load_agent_harness_source(ROOT, None)
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rwkv.jsonl"
            output_path = Path(tmp) / "predictions.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__project-1",
                        "output": f"<patch>\n{patch}</patch>",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(
                benchmark="swe_bench_verified",
                input=str(input_path),
                output=str(output_path),
                model="g1d-0.4b",
                target="auto",
                allow_empty_patch=False,
                allow_invalid=False,
            )

            result = agent_harness.convert_agent_outputs(source, root=ROOT, args=args)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["target"], "swebench-predictions")
        self.assertEqual(result["written"], 1)
        self.assertEqual(rows[0]["instance_id"], "repo__project-1")
        self.assertEqual(rows[0]["model_patch"], patch)

    def test_agent_format_reads_multirow_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"instance_id": "repo__project-1", "output": "first"}),
                        json.dumps({"instance_id": "repo__project-2", "output": "second"}),
                    ]
                ),
                encoding="utf-8",
            )

            records = agent_format.read_json_records(path)

        self.assertEqual([record["instance_id"] for record in records], ["repo__project-1", "repo__project-2"])

    def test_agent_format_intermediate_rows_keep_patch_artifact_and_metadata(self) -> None:
        patch = "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"

        rows, errors = agent_format.canonical_intermediate_rows(
            [
                {
                    "task_id": "repo__project-1",
                    "model": "rwkv",
                    "output": patch,
                    "score": 0.0,
                }
            ],
            benchmark="swe_bench_verified",
            model="g1d-0.4b",
        )

        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["format"], agent_format.INTERMEDIATE_FORMAT)
        self.assertEqual(rows[0]["sample_id"], "repo__project-1")
        self.assertEqual(rows[0]["artifacts"]["patch"], patch)
        self.assertEqual(rows[0]["metadata"]["score"], 0.0)

    def test_scoreboard_domain_classifies_agent_scope_narrowly(self) -> None:
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        from scoreboard_server.cores.normalize import domain_for

        self.assertEqual(domain_for("swe_bench_verified"), "agent")
        self.assertEqual(domain_for("browsecomp"), "agent")
        self.assertEqual(domain_for("mcp_bench_multi_2server"), "agent")
        self.assertEqual(domain_for("tau3_bench_mock"), "agent")
        self.assertEqual(domain_for("bfcl_v3"), "function_call")
        self.assertEqual(domain_for("toolalpaca_eval_real"), "function_call")
        self.assertEqual(domain_for("gpqa_diamond"), "knowledge")
        self.assertEqual(domain_for("cl_bench"), "knowledge")
        self.assertEqual(domain_for("matharena_apex"), "math")

    def test_scoreboard_leaderboard_uses_db_catalog_field_before_heuristics(self) -> None:
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        from scoreboard_server.cores.leaderboard import build_leaderboard_payload

        official_entry = {
            "score_id": 1,
            "task_id": 1,
            "cot": False,
            "cot_mode": "NoCoT",
            "metrics": {"accuracy": 0.5},
            "created_at": "2026-07-17T00:00:00",
            "is_param_search": False,
            "model": "rwkv7-g1g-1.5b",
            "dataset": "mmlu:machine_learning",
            "samples": 1,
            "problems": 1,
            "task": "lighteval",
            "task_details": None,
            "sampling_config": None,
            "log_path": "",
            "field": "coding",
        }
        tuning_entry = {
            **official_entry,
            "score_id": 2,
            "task_id": 2,
            "metrics": {"accuracy": 0.6},
            "is_param_search": True,
            "cot": True,
            "cot_mode": "CoT",
            "sampling_config": {
                "prompt_profile": "normal",
                "sampling_config": {"answer": {"temperature": 0.7, "top_k": 40, "top_p": 0.9}},
            },
        }
        payload = build_leaderboard_payload(
            [official_entry],
            selected_model=None,
            view="benchmark_detail_latest",
            tuning_entries=[tuning_entry],
        )

        rows_by_domain = {domain["key"]: domain["rows"] for domain in payload["domains"]}
        self.assertEqual(rows_by_domain["coding"][0]["benchmark_name"], "mmlu:machine_learning")
        self.assertEqual(rows_by_domain["knowledge"], [])
        matrix = next(domain for domain in payload["matrix"]["domains"] if domain["key"] == "coding")
        self.assertEqual(matrix["columns"][0]["key"], "mmlu:machine_learning")
        self.assertEqual(matrix["rows"][0]["average"], 50.0)
        tuning = payload["tuning_matrix"]["benchmarks"][0]
        self.assertEqual(tuning["columns"][0]["label"], "normal · CoT · T0.7 · K40 · P0.9")
        self.assertEqual(tuning["rows"][0]["best"], 60.0)

    def test_lighteval_tasks_judges_classifies_builtin_and_custom_metrics(self) -> None:
        class UpstreamAvgAtN:
            pass

        UpstreamAvgAtN.__module__ = "lighteval.metrics.metrics_sample"

        class FakeRegistry:
            _task_registry = {
                "aime24": SimpleNamespace(
                    metrics=(
                        SimpleNamespace(metric_name="pass@k:k=1", sample_level_fn=UpstreamAvgAtN()),
                        SimpleNamespace(metric_name="avg@n:n=1", sample_level_fn=UpstreamAvgAtN()),
                    )
                ),
                "tau3_bench_mock": SimpleNamespace(
                    metrics=(
                        lighteval_rwkv_skills_tasks.rwkv_tau_bench_static_plan_f1,
                        lighteval_rwkv_skills_tasks.rwkv_tau_bench_response_nonempty,
                    )
                ),
            }
            _task_superset_dict = {"demo_family": ("aime24", "tau3_bench_mock")}

        with mock.patch.object(lighteval_tasks, "load_registry", return_value=FakeRegistry()):
            rows = lighteval_tasks.judge_rows(
                Namespace(
                    custom_tasks=None,
                    load_multilingual=False,
                    tasks="demo_family",
                    contains=None,
                    limit=None,
                )
            )

        by_metric = {row.metric: row for row in rows}
        self.assertEqual(by_metric["avg@n:n=1"].source, "lighteval_builtin")
        self.assertEqual(by_metric["avg@n:n=1"].status, "ready")
        self.assertEqual(by_metric["avg@n:n=1"].judge_type, "avg_at_n")
        self.assertEqual(by_metric["tau_bench_static_plan_f1"].source, "helicopter_custom")
        self.assertEqual(by_metric["tau_bench_static_plan_f1"].status, "proxy")
        self.assertEqual(by_metric["tau_bench_response_nonempty"].status, "sanity")
        summary = lighteval_tasks.format_judges(rows, "summary")
        self.assertIn("tasks\t2\n", summary)
        self.assertIn("status\tready\t2\n", summary)
        self.assertIn("status\tproxy\t1\n", summary)
        self.assertIn("status\tsanity\t1\n", summary)

    def test_supergpqa_custom_task_prompt_keeps_all_options(self) -> None:
        doc = lighteval_rwkv_skills_tasks.supergpqa_prompt(
            {
                "question": "Pick the third option.",
                "options": ["zero", "one", "two", "three", "four"],
                "answer_letter": "C",
            },
            "supergpqa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.gold_index, 2)
        self.assertEqual(doc.choices, [" A", " B", " C", " D", " E"])
        self.assertIn("E. four", doc.query)

    def test_rwkv_skills_custom_tasks_include_direct_math_and_code_ids(self) -> None:
        registered_tasks = {task.name for task in lighteval_rwkv_skills_tasks.TASKS_TABLE}
        self.assertTrue(
            {
                "algebra222",
                "amc23",
                "answer_judge",
                "arena_hard_v2",
                "agentbench_db",
                "agentbench_kg",
                "beyond_aime",
                "brumo25",
                "browsecomp",
                "browsecomp_plus",
                "browsecomp_zh",
                "college_math",
                "comp_math_24_25",
                "gaokao2023en",
                "hendrycks_math",
                "hmmt_feb25",
                "human_eval",
                "human_eval_cn",
                "human_eval_fix",
                "human_eval_plus",
                "longbench",
                "longbench_qa",
                "longcodeqa",
                "mcp_bench",
                "mcp_bench_multi_2server",
                "mcp_bench_multi_3server",
                "mcp_bench_single",
                "math_odyssey",
                "mawps",
                "mbpp",
                "mbpp_plus",
                "minerva_math",
                "omni_math",
                "polymath",
                "svamp",
                "swe_bench",
                "swe_bench_lite",
                "swe_bench_verified",
                "swe_bench_lite_oracle",
                "swe_bench_lite_bm25_13k",
                "supergpqa",
                "tau_bench_retail",
                "tau_bench_airline",
                "tau_bench_telecom",
                "tau2_bench_retail",
                "tau2_bench_airline",
                "tau2_bench_telecom",
                "tau3_bench_retail",
                "tau3_bench_airline",
                "tau3_bench_telecom",
                "tau3_bench_banking_knowledge",
                "tau3_bench_mock",
                "tau3_bench_mock_long_context",
                "wmt24pp",
            }.issubset(registered_tasks)
        )
        self.assertFalse(
            {
                "apibank_l1",
                "apibank_l2",
                "apibank_level1",
                "apibank_level2",
                "bfcl_multiple",
                "bfcl_exec_multiple",
                "bfcl_exec_multiple_ast",
                "bfcl_exec_parallel",
                "bfcl_exec_parallel_multiple",
                "bfcl_simple_python",
                "bfcl_exec_simple",
                "bfcl_exec_simple_ast",
                "bfcl_v3",
                "complexfuncbench_official",
                "complexfuncbench_subset",
                "longbench_qa_balanced",
                "toolalpaca_eval_real",
                "toolalpaca_eval_simulated",
            }
            & registered_tasks
        )

    def test_polymath_task_aggregates_all_languages_and_levels(self) -> None:
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_LANGUAGES), 18)
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_LEVELS), 4)
        self.assertEqual(len(lighteval_rwkv_skills_tasks.POLYMATH_URLS), 72)
        self.assertIn("zh/low.parquet", lighteval_rwkv_skills_tasks.POLYMATH_URLS[-1])

    def test_comp_math_static_data_file_is_packaged(self) -> None:
        self.assertTrue(Path(lighteval_rwkv_skills_tasks.COMP_MATH_24_25_PATH).is_file())

    def test_minerva_math_static_data_file_is_packaged(self) -> None:
        path = Path(lighteval_rwkv_skills_tasks.MINERVA_MATH_PATH)
        self.assertTrue(path.is_file())
        with path.open(encoding="utf-8") as fh:
            self.assertEqual(sum(1 for _line in fh), 272)

    def test_dataset_resilience_retries_only_transient_errors(self) -> None:
        transient = ConnectionError("proxy disconnected")
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise transient
            return "ok"

        with mock.patch.dict(
            os.environ,
            {
                "HELICOPTER_LIGHTEVAL_DATASET_RETRIES": "2",
                "HELICOPTER_LIGHTEVAL_DATASET_RETRY_DELAY": "0",
            },
        ):
            self.assertEqual(lighteval_dataset_resilience.retry_load_dataset(flaky), "ok")
        self.assertEqual(calls, 3)
        self.assertFalse(lighteval_dataset_resilience.is_transient_dataset_error(ValueError("bad data")))

    def test_dataset_resilience_fetches_only_an_offline_cache_miss(self) -> None:
        calls = 0

        def cached_then_online() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("OfflineModeIsEnabled: dataset is not cached")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "0")
            return "downloaded"

        with mock.patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HELICOPTER_LIGHTEVAL_DATASET_ONLINE_FALLBACK": "1",
                "HELICOPTER_LIGHTEVAL_DATASET_RETRIES": "0",
            },
            clear=False,
        ):
            self.assertEqual(
                lighteval_dataset_resilience.retry_load_dataset(cached_then_online),
                "downloaded",
            )
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(calls, 2)

    def test_online_fallback_resets_huggingface_sessions(self) -> None:
        reset_sessions = mock.Mock()
        with mock.patch.object(
            lighteval_dataset_resilience, "_reset_huggingface_sessions", reset_sessions
        ):
            with lighteval_dataset_resilience._temporary_online_dataset_access():
                pass
        self.assertEqual(reset_sessions.call_count, 2)

    def test_mcpbench_static_data_files_are_packaged(self) -> None:
        expected_counts = {
            "mcp_bench": 104,
            "mcp_bench_single": 56,
            "mcp_bench_multi_2server": 30,
            "mcp_bench_multi_3server": 18,
        }
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.MCP_BENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_agentbench_static_data_files_are_packaged(self) -> None:
        expected_counts = {"agentbench_db": 300, "agentbench_kg": 150}
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.AGENTBENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_taubench_static_data_files_are_packaged(self) -> None:
        expected_counts = {
            "tau_bench_retail": 40,
            "tau_bench_airline": 20,
            "tau_bench_telecom": 40,
            "tau2_bench_retail": 114,
            "tau2_bench_airline": 50,
            "tau2_bench_telecom": 114,
            "tau3_bench_retail": 114,
            "tau3_bench_airline": 50,
            "tau3_bench_telecom": 114,
            "tau3_bench_banking_knowledge": 97,
            "tau3_bench_mock": 3,
            "tau3_bench_mock_long_context": 2,
        }
        self.assertEqual(set(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS), set(expected_counts))
        for name, expected_count in expected_counts.items():
            path = Path(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS[name])
            self.assertTrue(path.is_file(), name)
            with path.open(encoding="utf-8") as fh:
                self.assertEqual(sum(1 for _line in fh), expected_count, name)

    def test_taubench_mock_update_row_exposes_update_tool(self) -> None:
        path = Path(lighteval_rwkv_skills_tasks.TAU_BENCH_PATHS["tau3_bench_mock"])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        update_row = next(row for row in rows if row["task_id"] == "update_task_with_history_and_env_assertions")
        self.assertIn("update_task_status", update_row["available_action_names"])
        self.assertEqual(update_row["reference_action_names"], ["update_task_status"])
        self.assertIn("update_task_status", update_row["reference_plan"])

    def test_wmt24pp_task_uses_default_target_languages(self) -> None:
        self.assertEqual(lighteval_rwkv_skills_tasks.WMT24PP_TARGET_LANGUAGES, ("de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"))
        self.assertEqual(len(lighteval_rwkv_skills_tasks.WMT24PP_URLS), 5)
        self.assertIn("en-ja_JP.jsonl", lighteval_rwkv_skills_tasks.WMT24PP_URLS[-1])

    def test_wmt24pp_prompt_builds_translation_doc(self) -> None:
        doc = lighteval_rwkv_skills_tasks.wmt24pp_prompt(
            {
                "lp": "en-de_DE",
                "source": "Good morning.",
                "target": "Guten Morgen.",
            },
            "wmt24pp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["Guten Morgen."])
        self.assertIn("from English to German", doc.query)
        self.assertTrue(doc.query.endswith("German:"))

    def test_longbench_prompt_builds_references_and_truncates_context(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longbench_prompt(
            {
                "dataset": "triviaqa",
                "input": "What is the answer?",
                "context": "A" * (lighteval_rwkv_skills_tasks.LONG_CONTEXT_PROMPT_MAX_CHARS + 1000),
                "answers": ["forty two", "42"],
            },
            "longbench_qa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["references"], ["forty two", "42"])
        self.assertIn("middle truncated", doc.query)
        self.assertLess(len(doc.query), lighteval_rwkv_skills_tasks.LONG_CONTEXT_PROMPT_MAX_CHARS + 1000)

    def test_longbench_metric_scores_json_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longbench_prompt(
            {
                "input": "Name it.",
                "context": "The answer is Ozalj.",
                "answers": ["Ozalj"],
            },
            "longbench_qa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        exact = lighteval_rwkv_skills_tasks.LongBenchExactMatch()
        f1 = lighteval_rwkv_skills_tasks.LongBenchF1()
        response = ModelResponse(text=['{"answer":"Ozalj"}'])
        self.assertEqual(exact.compute(response, doc), 1.0)
        self.assertEqual(f1.compute(response, doc), 1.0)

    def test_longcodeqa_prompt_and_metric_accept_option_letter(self) -> None:
        doc = lighteval_rwkv_skills_tasks.longcodeqa_prompt(
            {
                "prompt": "Repository: Repository:\nexample\nQuestion:\nA) no\nB) yes\n",
                "question": "Question:\nA) no\nB) yes\n",
                "correct_letter": "B",
            },
            "longcodeqa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Repository:\nexample", doc.query)
        self.assertNotIn("Repository: Repository:", doc.query)
        self.assertEqual(doc.specific["correct_letter"], "B")
        self.assertEqual(doc.specific["allowed_letters"], ["A", "B"])
        metric = lighteval_rwkv_skills_tasks.LongCodeQAAccuracy()
        self.assertEqual(metric.compute(ModelResponse(text=['{"arguments":{"answer":"B"}}']), doc), 1.0)
        self.assertEqual(metric.compute(ModelResponse(text=["Answer: A"]), doc), 0.0)

    def test_browsecomp_prompt_decrypts_openai_csv_row(self) -> None:
        canary = "unit-canary"
        question = "Which city hosted the example event?"
        answer = "Ozalj"
        doc = lighteval_rwkv_skills_tasks.browsecomp_prompt(
            {
                "problem": self._browsecomp_encrypt(question, canary),
                "answer": self._browsecomp_encrypt(answer, canary),
                "problem_topic": "Geography",
                "canary": canary,
            },
            "browsecomp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(question, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["locale"], "en")
        metric = lighteval_rwkv_skills_tasks.BrowseCompExactMatch()
        self.assertEqual(metric.compute(ModelResponse(text=["Explanation: short\nExact Answer: Ozalj\nConfidence: 90%"]), doc), 1.0)

    def test_browsecomp_zh_prompt_decrypts_hf_parquet_row(self) -> None:
        canary = "BrowseComp-ZH"
        question = "这个示例问题的答案是什么？"
        answer = "示例答案"
        topic = "示例"
        doc = lighteval_rwkv_skills_tasks.browsecomp_prompt(
            {
                "Question": self._browsecomp_encrypt(question, canary),
                "Answer": self._browsecomp_encrypt(answer, canary),
                "Topic": self._browsecomp_encrypt(topic, canary),
                "canary": canary,
            },
            "browsecomp_zh",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(question, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["locale"], "zh")
        self.assertEqual(doc.specific["topic"], topic)
        metric = lighteval_rwkv_skills_tasks.BrowseCompF1()
        self.assertEqual(metric.compute(ModelResponse(text=["最终答案: 示例答案"]), doc), 1.0)

    def test_browsecomp_plus_prompt_decrypts_hf_row(self) -> None:
        canary = lighteval_rwkv_skills_tasks.BROWSECOMP_PLUS_CANARY
        query = "Which town hosts the example festival?"
        answer = "Ozalj"
        evidence = "The example festival is hosted in Ozalj every summer."
        doc = lighteval_rwkv_skills_tasks.browsecomp_plus_prompt(
            {
                "query_id": "unit-1",
                "query": self._browsecomp_encrypt(query, canary),
                "answer": self._browsecomp_encrypt(answer, canary),
                "gold_docs": [
                    {
                        "docid": self._browsecomp_encrypt("doc-1", canary),
                        "text": self._browsecomp_encrypt(evidence, canary),
                        "url": self._browsecomp_encrypt("https://example.test/doc", canary),
                    }
                ],
                "evidence_docs": [],
                "negative_docs": [],
            },
            "browsecomp_plus",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn(query, doc.query)
        self.assertIn(evidence, doc.query)
        self.assertEqual(doc.specific["references"], [answer])
        self.assertEqual(doc.specific["query_id"], "unit-1")
        self.assertEqual(doc.specific["mode"], "oracle_context")
        metric = lighteval_rwkv_skills_tasks.BrowseCompExactMatch()
        self.assertEqual(metric.compute(ModelResponse(text=["Exact Answer: Ozalj"]), doc), 1.0)

    def test_bfcl_prompt_scores_json_tool_call_against_ast_ground_truth(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_multiple_0",
                "question": [[{"role": "user", "content": "Compute a binomial probability."}]],
                "function": [
                    {
                        "name": "calc_binomial_probability",
                        "description": "Calculate binomial probability.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "execution_result_type": ["exact_match"],
                "ground_truth": ["calc_binomial_probability(n=20, k=5, p=1/6)"],
            },
            "bfcl_exec_multiple",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Available functions:", doc.query)
        self.assertEqual(doc.specific["sample_id"], "exec_multiple_0")
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=['{"name":"calc_binomial_probability","arguments":{"n":20,"k":5,"p":0.1666666667}}']
                ),
                doc,
            ),
            1.0,
        )

    def test_bfcl_metric_matches_parallel_calls_orderlessly(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_parallel_0",
                "question": [[{"role": "user", "content": "Run three probability calculations."}]],
                "function": [{"name": "calc_binomial_probability", "parameters": {"type": "object"}}],
                "ground_truth": [
                    "calc_binomial_probability(n=10, k=3, p=0.3)",
                    "calc_binomial_probability(n=15, k=5, p=0.3)",
                ],
            },
            "bfcl_exec_parallel",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        response = ModelResponse(
            text=[
                json.dumps(
                    [
                        {"name": "calc_binomial_probability", "arguments": {"n": 15, "k": 5, "p": 0.3}},
                        {"name": "calc_binomial_probability", "arguments": {"n": 10, "k": 3, "p": 0.3}},
                    ]
                )
            ]
        )
        self.assertEqual(metric.compute(response, doc), 1.0)

    def test_bfcl_metric_extracts_multiple_tool_call_blocks(self) -> None:
        doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
            {
                "id": "exec_parallel_asin",
                "question": [[{"role": "user", "content": "Get ratings for two products."}]],
                "function": [{"name": "get_rating_by_amazon_ASIN", "parameters": {"type": "object"}}],
                "ground_truth": [
                    "get_rating_by_amazon_ASIN(ASIN='B08PPDJWC8')",
                    "get_rating_by_amazon_ASIN(ASIN='B07ZPKBL9V')",
                ],
            },
            "bfcl_exec_parallel",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        response = ModelResponse(
            text=[
                "<tool_call>\n"
                '{"name":"get_rating_by_amazon_ASIN","arguments":"{\\"ASIN\\":\\"B08PPDJWC8\\"}"}'
                "\n</tool_call>\n"
                "<tool_call>\n"
                '{"name":"get_rating_by_amazon_ASIN","arguments":"{\\"ASIN\\":\\"B07ZPKBL9V\\"}"}'
                "\n</tool_call>"
            ]
        )
        self.assertEqual(metric.compute(response, doc), 1.0)

    def test_bfcl_prompt_joins_possible_answer_by_id(self) -> None:
        with mock.patch.object(
            lighteval_rwkv_skills_tasks,
            "_bfcl_possible_answers",
            return_value={
                "simple_0": [
                    {
                        "calculate_triangle_area": {
                            "base": [10],
                            "height": [5],
                            "unit": ["units", ""],
                        }
                    }
                ]
            },
        ):
            doc = lighteval_rwkv_skills_tasks.bfcl_prompt(
                {
                    "id": "simple_0",
                    "question": [[{"role": "user", "content": "Find the triangle area."}]],
                    "function": [{"name": "calculate_triangle_area", "parameters": {"type": "object"}}],
                },
                "bfcl_simple_python",
            )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["sample_id"], "simple_0")
        self.assertNotIn("expected_calls", doc.specific)
        self.assertIn("calculate_triangle_area", doc.specific["expected_calls_json"])
        metric = lighteval_rwkv_skills_tasks.BFCLAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=['{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}']),
                doc,
            ),
            1.0,
        )

    def test_apibank_prompt_scores_against_sandbox_result(self) -> None:
        expected_result = {"input": {"city": "Paris"}, "output": {"weather": "sunny"}, "exception": None}
        doc = lighteval_rwkv_skills_tasks.apibank_prompt(
            {
                "task_id": "apibank_level1__weather_001",
                "instruction": "User: What is the weather in Paris?",
                "tools_json": json.dumps(
                    [
                        {
                            "name": "GetWeather",
                            "description": "Get weather.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        }
                    ]
                ),
                "expected_call_json": json.dumps({"name": "GetWeather", "arguments": {"city": "Paris"}}),
                "expected_result_json": json.dumps(expected_result),
            },
            "apibank_level1",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("API-Bank", doc.query)
        self.assertEqual(doc.specific["sample_id"], "apibank_level1__weather_001")
        self.assertNotIn("expected_tool_calls", doc.specific)

        case = self

        class FakeSandbox:
            def replay_history(self, source_path, turn_index):
                case.assertEqual(source_path, "")
                case.assertEqual(turn_index, 1)

            def api_call(self, name, arguments):
                case.assertEqual(name, "GetWeather")
                case.assertEqual(arguments, {"city": "Paris"})
                return lighteval_rwkv_skills_tasks.ApiBankCallResult(True, expected_result)

            def check_api_call_correctness(self, name, actual, expected):
                case.assertEqual(name, "GetWeather")
                return actual == expected

            def _api_info(self, name):
                case.assertEqual(name, "GetWeather")
                return {"input_parameters": {"city": {"type": "str"}}}

            @staticmethod
            def _coerce_arg(value, arg_type):
                return value

        with mock.patch.object(lighteval_rwkv_skills_tasks, "ApiBankSandbox", return_value=FakeSandbox()):
            metric = lighteval_rwkv_skills_tasks.APIBankAccuracy()
            self.assertEqual(
                metric.compute(
                    ModelResponse(text=['{"name":"GetWeather","arguments":{"city":"Paris"}}']),
                    doc,
                ),
                1.0,
            )

    def test_apibank_metric_accepts_gold_arguments_when_official_execution_fails(self) -> None:
        expected_call = {
            "name": "CancelTimedSwitch",
            "arguments": {"device_id": "10000025", "time": "2023-03-19 09:30:00"},
        }
        doc = lighteval_rwkv_skills_tasks.apibank_prompt(
            {
                "task_id": "apibank_level1__CancelTimedSwitch-level-1-1_002",
                "instruction": "User: Cancel the timed switch.",
                "tools_json": json.dumps(
                    [
                        {
                            "name": "CancelTimedSwitch",
                            "description": "Cancels a timed switch.",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                ),
                "expected_call_json": json.dumps(expected_call),
                "expected_result_json": json.dumps(
                    {
                        "api_name": "CancelTimedSwitch",
                        "input": {"device_id": "10000025", "time": "2023-03-19 09:30:00"},
                        "output": "success",
                        "exception": None,
                    }
                ),
                "source_path": "CancelTimedSwitch-level-1-1.jsonl",
                "turn_index": 2,
            },
            "apibank_level1",
        )
        self.assertIsNotNone(doc)
        assert doc is not None

        case = self

        class FakeSandbox:
            def replay_history(self, source_path, turn_index):
                case.assertEqual(source_path, "CancelTimedSwitch-level-1-1.jsonl")
                case.assertEqual(turn_index, 2)

            def api_call(self, name, arguments):
                case.assertEqual(name, "CancelTimedSwitch")
                case.assertEqual(arguments, {"name": "10000025", "time": "2023-03-19 09:30:00"})
                return lighteval_rwkv_skills_tasks.ApiBankCallResult(False, error="device name does not exist.")

            def _api_info(self, name):
                case.assertEqual(name, "CancelTimedSwitch")
                return {"input_parameters": {"name": {"type": "str"}, "time": {"type": "str"}}}

            @staticmethod
            def _coerce_arg(value, arg_type):
                return value

        with mock.patch.object(lighteval_rwkv_skills_tasks, "ApiBankSandbox", return_value=FakeSandbox()):
            metric = lighteval_rwkv_skills_tasks.APIBankAccuracy()
            self.assertEqual(
                metric.compute(
                    ModelResponse(
                        text=[
                            '{"name":"CancelTimedSwitch","arguments":'
                            '{"name":"10000025","time":"2023-03-19 09:30:00"}}'
                        ]
                    ),
                    doc,
                ),
                1.0,
            )

    def test_apibank_checker_falls_back_when_official_checker_raises(self) -> None:
        class RaisingTool:
            def check_api_call_correctness(self, actual, expected):
                raise KeyError("time")

        class FakeSandbox(lighteval_rwkv_skills_tasks.ApiBankSandbox):
            def __init__(self):
                pass

            def init_tool(self, api_name):
                return RaisingTool()

        actual = {
            "api_name": "DeleteMeeting",
            "input": {
                "attendees": ["David Wang", "Amy Chen"],
                "end_time": "2023-03-27 11:00:00",
                "location": "Training Room",
                "meeting_topic": "New Employee Orientation",
                "start_time": "2023-03-27 09:00:00",
                "token": "token",
            },
            "output": "success",
            "exception": None,
        }
        self.assertTrue(FakeSandbox().check_api_call_correctness("DeleteMeeting", actual, copy.deepcopy(actual)))
        wrong = copy.deepcopy(actual)
        wrong["input"]["location"] = "Other Room"
        self.assertFalse(FakeSandbox().check_api_call_correctness("DeleteMeeting", actual, wrong))

    def test_toolalpaca_prompt_scores_matching_request(self) -> None:
        doc = lighteval_rwkv_skills_tasks.toolalpaca_prompt(
            {
                "task_id": "toolalpaca_eval_simulated__weather_000",
                "instruction": "Find the current weather in Paris.",
                "tools": [
                    {
                        "name": "getWeather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        "metadata": {
                            "path": "/weather/{city}",
                            "method": "get",
                            "operation": {
                                "parameters": [
                                    {
                                        "name": "city",
                                        "in": "path",
                                        "required": True,
                                        "schema": {"type": "string"},
                                    }
                                ]
                            },
                        },
                    }
                ],
                "expected_tool_calls": [
                    {
                        "name": "getWeather",
                        "arguments": {"city": "Paris"},
                        "argument_options": {"city": ["Paris"]},
                    }
                ],
            },
            "toolalpaca_eval_simulated",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("ToolAlpaca", doc.query)
        self.assertNotIn('"metadata"', doc.query)
        self.assertEqual(doc.specific["sample_id"], "toolalpaca_eval_simulated__weather_000")
        metric = lighteval_rwkv_skills_tasks.ToolAlpacaAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=['{"name":"getWeather","arguments":{"city":"Paris"}}']),
                doc,
            ),
            1.0,
        )

    def test_complexfuncbench_prompt_scores_matching_parallel_call_turn(self) -> None:
        doc = lighteval_rwkv_skills_tasks.complexfuncbench_prompt(
            {
                "id": "complex-case-1",
                "functions": [
                    {
                        "name": "SearchHotel",
                        "description": "Search hotels.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "BookHotel",
                        "description": "Book hotels.",
                        "parameters": {
                            "type": "object",
                            "properties": {"hotel_id": {"type": "string"}},
                            "required": ["hotel_id"],
                        },
                    },
                ],
                "conversations": [
                    {"role": "user", "content": "Find and book h1 in Paris."},
                    {
                        "role": "assistant",
                        "function_call": [
                            {"name": "SearchHotel", "arguments": {"city": "Paris"}},
                            {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
                        ],
                    },
                    {"role": "observation", "content": [{"hotel_id": "h1"}, {"status": "booked"}]},
                ],
            },
            "complexfuncbench_official",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("ComplexFuncBench", doc.query)
        self.assertIn("SearchHotel", doc.query)
        self.assertEqual(doc.specific["sample_id"], "complex-case-1__turn_1")
        metric = lighteval_rwkv_skills_tasks.ComplexFuncBenchCallAccuracy()
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=[
                        '[{"name":"SearchHotel","arguments":{"city":"Paris"}},'
                        '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}]'
                    ]
                ),
                doc,
            ),
            1.0,
        )
        self.assertEqual(
            metric.compute(
                ModelResponse(
                    text=[
                        '[{"name":"BookHotel","arguments":{"hotel_id":"h1"}},'
                        '{"name":"SearchHotel","arguments":{"city":"Paris"}}]'
                    ]
                ),
                doc,
            ),
            0.0,
        )

    def test_arena_hard_prompt_scores_against_baseline_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.arena_hard_prompt(
            {
                "uid": "arena-1",
                "prompt": "Explain why the sky appears blue.",
                "baseline_answer": "The sky appears blue because air molecules scatter shorter blue wavelengths.",
            },
            "arena_hard_v2",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Arena-Hard", doc.query)
        self.assertEqual(doc.specific["sample_id"], "arena-1")
        metric = lighteval_rwkv_skills_tasks.ArenaHardBaselineF1()
        self.assertEqual(
            metric.compute(
                ModelResponse(text=["The sky appears blue because air molecules scatter shorter blue wavelengths."]),
                doc,
            ),
            1.0,
        )

    def test_swebench_prompt_scores_gold_patch_and_strips_context_patch(self) -> None:
        patch = "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
        doc = lighteval_rwkv_skills_tasks.swebench_prompt(
            {
                "instance_id": "repo__project-1",
                "repo": "repo/project",
                "base_commit": "abc123",
                "problem_statement": "Fix the example bug.",
                "hints_text": "Look at example.py.",
                "text": f"Relevant context before patch.\n<patch>\n{patch}</patch>\nDo not leak this.",
                "patch": f"<patch>\n{patch}</patch>",
            },
            "swe_bench_lite_oracle",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, "Fix the example bug.")
        self.assertNotIn("Relevant context before patch.", doc.query)
        self.assertNotIn("<patch>", doc.query)
        self.assertNotIn("-old", doc.query)
        self.assertEqual(doc.specific["sample_id"], "repo__project-1")
        self.assertEqual(doc.specific["harness_dataset_name"], "princeton-nlp/SWE-bench_Lite")

        f1 = lighteval_rwkv_skills_tasks.SweBenchPatchF1()
        nonempty = lighteval_rwkv_skills_tasks.SweBenchPatchNonEmpty()
        response = ModelResponse(text=[f"```diff\n{patch}```"])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=["I cannot produce a patch."]), doc), 0.0)

    def test_mcpbench_prompt_scores_static_plan(self) -> None:
        doc = lighteval_rwkv_skills_tasks.mcpbench_prompt(
            {
                "task_id": "weather_data_000",
                "instruction": "Find the hourly forecast for Seattle using only Weather Data.",
                "task_file": "mcpbench_tasks_single_runner_format.json",
                "server_name": "Weather Data",
                "servers": ["Weather Data"],
                "combination_name": "Single Server: Weather Data",
                "combination_type": "single_server",
                "official_source": "Accenture/mcp-bench",
                "official_source_revision": "revision",
                "official_source_path": "tasks/mcpbench_tasks_single_runner_format.json",
                "task": {
                    "task_id": "weather_data_000",
                    "task_description": "Use Weather Data:getForecast with latitude and longitude for Seattle.",
                    "dependency_analysis": "Call geocoding first, then use the coordinates in Weather Data:getForecast.",
                },
            },
            "mcp_bench_single",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("MCP-Bench", doc.query)
        self.assertIn("Weather Data", doc.query)
        self.assertEqual(doc.specific["sample_id"], "weather_data_000")
        self.assertEqual(doc.specific["servers"], ["Weather Data"])

        f1 = lighteval_rwkv_skills_tasks.McpBenchStaticPlanF1()
        nonempty = lighteval_rwkv_skills_tasks.McpBenchResponseNonEmpty()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_agentbench_db_prompt_scores_final_answer(self) -> None:
        doc = lighteval_rwkv_skills_tasks.agentbench_db_prompt(
            {
                "task_id": "agentbench_db__00000",
                "task_name": "dbbench-std",
                "index": 0,
                "domain": "dbbench",
                "question": "What are the Notes when the Method is decision?",
                "additional_description": "The table is Jiu-Jitsu Championships Results.",
                "operation_type": "other",
                "tables": [
                    {
                        "table_name": "Jiu-Jitsu Championships Results",
                        "columns": ["Method", "Notes"],
                        "rows": [["Decision", "Women +60kg Bronze"]],
                    }
                ],
                "reference_answers": ["Women +60kg Bronze"],
                "reference_sql": "SELECT Notes FROM table WHERE Method = 'Decision';",
                "reference_plan": "SQL: SELECT Notes FROM table WHERE Method = 'Decision';\nFinal answer: [\"Women +60kg Bronze\"]",
            },
            "agentbench_db",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("AgentBench DBBench", doc.query)
        self.assertIn("execute", doc.query.lower())
        self.assertEqual(doc.specific["sample_id"], "agentbench_db__00000")

        f1 = lighteval_rwkv_skills_tasks.AgentBenchDbAnswerF1()
        nonempty = lighteval_rwkv_skills_tasks.AgentBenchResponseNonEmpty()
        self.assertEqual(f1.compute(ModelResponse(text=['{"final_answer":"Women +60kg Bronze"}']), doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=['{"final_answer":"Women +60kg Bronze"}']), doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_agentbench_kg_prompt_scores_reference_plan(self) -> None:
        doc = lighteval_rwkv_skills_tasks.agentbench_kg_prompt(
            {
                "task_id": "agentbench_kg__00000",
                "task_name": "kg-std",
                "index": 0,
                "domain": "knowledgegraph",
                "question": "what is the attitude of the first dog and the german shepherds?",
                "entities": {"first dog": "m.05t073s", "german shepherds": "m.0km5c"},
                "reference_actions": ["get_relations(m.05t073s)", "intersection(#1,#2)"],
                "reference_answers": ["Obedient", "Intelligent"],
                "reference_plan": "Actions:\nget_relations(m.05t073s)\nintersection(#1,#2)\nFinal answer names: [\"Obedient\", \"Intelligent\"]",
            },
            "agentbench_kg",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("AgentBench KnowledgeGraph", doc.query)
        self.assertIn("get_relations", doc.query)
        self.assertEqual(doc.specific["sample_id"], "agentbench_kg__00000")

        f1 = lighteval_rwkv_skills_tasks.AgentBenchKgPlanF1()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)

    def test_taubench_prompt_scores_static_plan_without_leaking_criteria(self) -> None:
        reference_plan = json.dumps(
            {
                "actions": [{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Important Meeting"}}],
                "env_assertions": [{"func_name": "assert_task_status"}],
                "nl_assertions": [],
                "reward_basis": ["DB", "ENV_ASSERTION"],
            },
            sort_keys=True,
        )
        doc = lighteval_rwkv_skills_tasks.taubench_prompt(
            {
                "sample_id": "tau3_bench_mock__create_task_1",
                "task_id": "create_task_1",
                "domain": "mock",
                "split": "base",
                "benchmark_version": "tau_v3_light",
                "instruction": "Create a task named Important Meeting for user_1.",
                "available_action_names": ["create_task", "update_task_status"],
                "reference_action_names": ["create_task"],
                "reference_actions": [{"name": "create_task", "arguments": {"user_id": "user_1", "title": "Important Meeting"}}],
                "reference_plan": reference_plan,
                "task": {
                    "id": "create_task_1",
                    "ticket": "Create a task named Important Meeting for user_1.",
                    "evaluation_criteria": {"actions": [{"name": "create_task"}]},
                },
                "official_source": "https://github.com/sierra-research/tau2-bench",
                "official_source_revision": "revision",
                "official_source_path": "data/tau2/domains/mock",
            },
            "tau3_bench_mock",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("TAU benchmark", doc.query)
        self.assertIn("create_task", doc.query)
        self.assertNotIn("evaluation_criteria", doc.query)
        self.assertEqual(doc.specific["sample_id"], "tau3_bench_mock__create_task_1")
        self.assertEqual(doc.specific["reference_action_names"], ["create_task"])

        f1 = lighteval_rwkv_skills_tasks.TauBenchStaticPlanF1()
        nonempty = lighteval_rwkv_skills_tasks.TauBenchResponseNonEmpty()
        response = ModelResponse(text=[doc.specific["reference_plans"][0]])
        self.assertEqual(f1.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(response, doc), 1.0)
        self.assertEqual(nonempty.compute(ModelResponse(text=[""]), doc), 0.0)

    def test_lighteval_export_prefers_specific_sample_id(self) -> None:
        row = lighteval_export.export_rows_from_frame(
            [
                {
                    "doc": {
                        "id": "doc-id",
                        "task_name": "swe_bench_lite|0",
                        "query": "Patch:",
                        "choices": [""],
                        "gold_index": 0,
                        "specific": {"sample_id": "repo__project-1"},
                    },
                    "metric": {"swebench_patch_nonempty": 1.0},
                    "model_response": {"text": ["--- a/x\n+++ b/x\n"]},
                }
            ]
        )[0]

        self.assertEqual(row["sample_id"], "repo__project-1")
        self.assertTrue(row["is_correct"])

    def test_is_correct_ignores_graded_f1_metrics(self) -> None:
        # A small nonzero proxy token-F1 is not a correct answer.
        self.assertIsNone(lighteval_export.is_correct({"swebench_patch_f1": 0.05}))
        self.assertIsNone(lighteval_export.is_correct({"longbench_f1": 0.4}))
        self.assertIsNone(lighteval_export.is_correct({"f1": 0.9}))

    def test_is_correct_uses_binary_signal_over_graded(self) -> None:
        # extractiveness (graded) must not override the binary extractive_match=0.0.
        self.assertFalse(
            lighteval_export.is_correct({"extractive_match": 0.0, "extractiveness": 0.83})
        )
        self.assertTrue(
            lighteval_export.is_correct({"extractive_match": 1.0, "extractiveness": 0.1})
        )

    def test_is_correct_handles_non_dict_metric(self) -> None:
        self.assertIsNone(lighteval_export.is_correct(None))
        self.assertIsNone(lighteval_export.is_correct([1.0]))

    def test_export_rows_tolerate_non_dict_columns(self) -> None:
        # doc/metric/model_response arriving as non-dict cells must not crash.
        rows = lighteval_export.export_rows_from_frame(
            [
                {
                    "doc": ["not", "a", "dict"],
                    "metric": None,
                    "model_response": "plain string",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["is_correct"])
        self.assertIsNone(rows[0]["task_name"])

    def test_free_answer_prompt_normalizes_numeric_answers(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "question": "What is 40 + 3?",
                "final_answer": 43.0,
            },
            "algebra222",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["43"])
        self.assertEqual(doc.query, "What is 40 + 3?")
        self.assertIsNone(doc.instruction)

    def test_free_answer_prompt_builds_svamp_problem(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "Body": "Each pack costs 76 dollars.",
                "Question": "How much after a 25 dollar discount?",
                "Answer": 51.0,
            },
            "svamp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Each pack costs 76 dollars. How much", doc.query)
        self.assertEqual(doc.choices, ["51"])

    def test_free_answer_prompt_extracts_math_odyssey_sparse_row(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "Problem_1": {
                    "question": "\\begin{problem}Compute $1+1$.\\end{problem}",
                    "answer": "$2$.",
                },
                "Problem_2": None,
            },
            "math_odyssey",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertIn("Compute $1+1$.", doc.query)
        self.assertEqual(doc.choices, ["2"])

    def test_free_answer_prompt_keeps_empty_gold_rows(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "This source row has no gold answer.",
                "answer": "",
            },
            "omni_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, [""])

    def test_free_answer_prompt_extracts_boxed_solution_when_answer_missing(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "Find x.",
                "solution": "Solving gives \\\\boxed{1.6}.",
            },
            "minerva_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["1.6"])

    def test_free_answer_prompt_extracts_nested_boxed_solution(self) -> None:
        doc = lighteval_rwkv_skills_tasks.free_answer_prompt(
            {
                "problem": "Find x.",
                "solution": "Solving gives \\\\boxed{\\\\frac{1}{2}}.",
            },
            "minerva_math",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["\\\\frac{1}{2}"])

    def test_famous120_open_qa_prompt_keeps_raw_question_and_alias_golds(self) -> None:
        doc = lighteval_rwkv_skills_tasks.famous120_open_qa_prompt(
            {
                "question": "What is George Rankin's occupation?",
                "possible_answers": '["politician", "political leader"]',
                "obj": "politician",
            },
            "popqa",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, "What is George Rankin's occupation?")
        self.assertEqual(doc.choices, ['["politician", "political leader"]'])
        self.assertIsNone(doc.instruction)

    def test_famous120_math_prompt_keeps_raw_question_and_final_gold(self) -> None:
        doc = lighteval_rwkv_skills_tasks.famous120_math_prompt(
            {"input": "Calculate 2 + 3.", "target": 5.0},
            "gsm_hard",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, "Calculate 2 + 3.")
        self.assertEqual(doc.choices, ["5"])
        self.assertIsNone(doc.instruction)

    def test_muldimif_prompt_and_official_constraint_metrics(self) -> None:
        line = {
            "id": "sample-1",
            "conversations": [
                {
                    "role": "user",
                    "content": "Include the keyword 'alpha' and end with a period.",
                }
            ],
            "constraints": [
                ["Content", "Keywords", "Must include the keyword 'alpha'"],
                [
                    "Content",
                    "Punctuation",
                    "Ending punctuation must be a period",
                ],
            ],
            "constraint_pattern": "Listing",
            "difficulty": "Level 1",
        }
        doc = lighteval_rwkv_skills_tasks.muldimif_prompt(line, "muldimif")

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, line["conversations"][0]["content"])
        reference = json.loads(doc.choices[0])
        self.assertEqual(reference["contract"], "instruction_constraints")
        self.assertEqual(len(reference["constraints"]), 2)
        self.assertEqual(doc.specific["sample_id"], "sample-1")

        strict = lighteval_rwkv_skills_tasks.MulDimIFStrict()
        per_constraint = (
            lighteval_rwkv_skills_tasks.MulDimIFConstraintAccuracy()
        )
        passed = ModelResponse(text=["alpha."])
        failed = ModelResponse(text=["beta!"])
        self.assertEqual(strict.compute(passed, doc), 1.0)
        self.assertEqual(per_constraint.compute(passed, doc), 1.0)
        self.assertEqual(strict.compute(failed, doc), 0.0)
        self.assertEqual(per_constraint.compute(failed, doc), 0.0)

    def test_gsm_symbolic_prompt_uses_only_final_answer_after_hashes(self) -> None:
        doc = lighteval_rwkv_skills_tasks.gsm_symbolic_prompt(
            {
                "question": "What is 40 + 2?",
                "answer": "First calculate it.\n#### 42",
            },
            "gsm_symbolic",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, "What is 40 + 2?")
        self.assertEqual(doc.choices, ["42"])

    def test_numglue_prompt_preserves_dataset_context_and_options(self) -> None:
        doc = lighteval_rwkv_skills_tasks.numglue_prompt(
            {
                "passage": "Ray recorded 7 interceptions; Eugene recorded 4.",
                "question": "How many more did Ray record?",
                "option1": "3",
                "option2": "11",
                "answer": "Option 1",
            },
            "numglue",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(
            doc.query,
            "Ray recorded 7 interceptions; Eugene recorded 4.\n"
            "How many more did Ray record?\nOption 1: 3\nOption 2: 11",
        )
        self.assertEqual(doc.choices, ["Option 1"])

    def test_compact_lighteval_doc_keeps_identity_and_arena_reference(self) -> None:
        compact = scoreboard_bridge._compact_lighteval_doc(
            {
                "id": "130",
                "task_name": "arena_hard_v2",
                "query": "large prompt already stored in stages",
                "choices": ["large duplicated choice"],
                "specific": {
                    "sample_id": "sample-130",
                    "references": ["reference answer"],
                    "private_tests": ["large duplicated tests"],
                },
            }
        )

        self.assertEqual(
            compact,
            {
                "id": "130",
                "task_name": "arena_hard_v2",
                "specific": {
                    "sample_id": "sample-130",
                    "references": ["reference answer"],
                },
            },
        )

        with mock.patch.object(
            scoreboard_bridge,
            "_sampling_config",
            return_value={"generation_size": 2048},
        ):
            payload = scoreboard_bridge._generation_payloads(
                task_name="g1h__arena_hard_v2|0",
                sample_index=0,
                doc={"id": "x", "query": "q"},
                response={"input": "q", "text": ["answer"]},
                repeat_indices=[0],
                generation_size=4096,
            )[0]
        self.assertEqual(payload["sampling_config"]["effective_generation_size"], 4096)

    def test_answer_judge_prompt_uses_mean_annotation_score(self) -> None:
        doc = lighteval_rwkv_skills_tasks.answer_judge_prompt(
            {
                "question": "What is 2 + 2?",
                "gt_answer": "4",
                "gen_answer": "four",
                "annotations": [{"score": "1"}, {"score": "0"}, {"score": "1"}],
            },
            "answer_judge",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.choices, ["Judgement: Yes"])
        self.assertIn("Return exactly `Judgement: Yes` or `Judgement: No`.", doc.query)

    def test_arena_hard_score_stage_reuses_database_references(self) -> None:
        payloads = [
            {
                "context": {
                    "agent_result": {
                        "doc": {
                            "id": "130",
                            "specific": {
                                "sample_id": "6225fbb8f3084d57852db56882e972ba",
                                "references": ["$20,000 decrease."],
                            },
                        }
                    }
                }
            }
        ]
        lighteval_rwkv_skills_tasks._arena_hard_baseline_answers.cache_clear()
        with (
            mock.patch.dict(
                os.environ,
                {"HELICOPTER_PIPELINE_STAGE": "score", "HELICOPTER_SCOREBOARD_TASK_ID": "26"},
            ),
            mock.patch("helicopter_cli.scoreboard_bridge.load_lighteval_generation", return_value=payloads),
            mock.patch.object(lighteval_rwkv_skills_tasks.urllib.request, "urlopen") as urlopen,
        ):
            answers = lighteval_rwkv_skills_tasks._arena_hard_baseline_answers()
        lighteval_rwkv_skills_tasks._arena_hard_baseline_answers.cache_clear()
        self.assertEqual(answers, {"6225fbb8f3084d57852db56882e972ba": "$20,000 decrease."})
        urlopen.assert_not_called()

    def test_human_eval_prompt_preserves_execution_specifics(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "def add_one(x):\n    ",
                "entry_point": "add_one",
                "test": "def check(candidate):\n    assert candidate(1) == 2",
            },
            "human_eval",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["code_kind"], "human_eval")
        self.assertEqual(doc.specific["entry_point"], "add_one")
        self.assertEqual(doc.query, "def add_one(x):")

    def test_mbpp_prompt_uses_base_test_list_for_base_task(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "prompt": "Write a function to add one.",
                "code": "def add_one(x):\n    return x + 1\n",
                "test_imports": "[]",
                "test_list": "['assert add_one(1) == 2']",
                "test": "raise AssertionError('plus test should not be used')",
            },
            "mbpp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.specific["code_kind"], "mbpp")
        self.assertEqual(doc.specific["entry_point"], "add_one")
        self.assertIn("assert add_one(1) == 2", doc.specific["test"])
        self.assertNotIn("plus test should not be used", doc.specific["test"])

    def test_mbpp_prompt_accepts_official_huggingface_schema(self) -> None:
        doc = lighteval_rwkv_skills_tasks.code_generation_prompt(
            {
                "text": "Write a Python function to add one.",
                "code": "def add_one(x):\n    return x + 1\n",
                "test_setup_code": "from math import floor",
                "test_list": ["assert add_one(1) == 2"],
            },
            "mbpp",
        )

        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.query, "Write a Python function to add one.")
        self.assertEqual(doc.specific["entry_point"], "add_one")
        self.assertIn("from math import floor", doc.specific["test"])
        self.assertIn("assert add_one(1) == 2", doc.specific["test"])

    def test_official_avg_at_n_calls_native_math_scorer_per_rollout(self) -> None:
        from lighteval.metrics.metrics import Metrics

        metric = Metrics.avg_at_n_math.value
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=2, name="avg@2")
        doc = Doc(query="Find the value.", choices=["68"], gold_index=0)
        response = ModelResponse(text=[r"\boxed{68}", r"\boxed{67}"])

        self.assertEqual(wrapped.sample_level_fn.compute(doc, response), 0.5)

    def test_avg_math_uses_the_canonical_answer_adapter_when_configured(self) -> None:
        from lighteval.metrics.metrics import Metrics

        metric = Metrics.avg_at_n_math.value
        wrapped = lighteval_g1h_policy._avg_metric(
            metric,
            k=2,
            name="avg@2",
            domain="math",
            request_format="math_boxed",
        )
        doc = Doc(query="Find the value.", choices=["$\\boxed{540}$"], gold_index=0)
        response = ModelResponse(
            text=[
                r"reasoning</think>The maximum is \(\sqrt{324^2 + 432^2}=540\).",
                r"\boxed{541}",
            ]
        )

        self.assertEqual(wrapped.sample_level_fn.compute(doc, response), 0.5)

    def test_request_policy_reads_the_launcher_environment_name(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY": json.dumps(
                    {"tasks": {"aime24": {"domain": "math", "format": "math_boxed"}}}
                ),
                "HELICOPTER_LIGHTEEVAL_TASK_REQUEST_POLICY": "",
            },
            clear=False,
        ):
            request_policy = lighteval_g1h_policy._request_policy_from_environment()

        self.assertEqual(request_policy["domain"], "math")
        self.assertEqual(request_policy["format"], "math_boxed")

    def test_avg_wrapper_uses_native_light_eval_math_metric(self) -> None:
        from lighteval.metrics.metrics_sample import ExactMatches

        metric = SimpleNamespace(
            metric_name="em",
            sample_level_fn=ExactMatches(strip_strings=True),
            category=lighteval_g1h_policy.SamplingMethod.GENERATIVE,
            corpus_level_fn=lambda values: sum(values) / len(values),
            higher_is_better=True,
        )
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=2, name="avg@2")
        self.assertIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(wrapped.sample_level_fn.n, 2)

    def test_avg_wrapper_replaces_project_scorer_with_official_metric(self) -> None:
        from lighteval.metrics.metrics import Metrics
        from lighteval.metrics.metrics_sample import SampleLevelComputation

        class ProjectScorer(SampleLevelComputation):
            def compute(self, doc, model_response, **kwargs):
                return 1.0

        metric = SimpleNamespace(
            metric_name="project_metric",
            sample_level_fn=ProjectScorer(),
            category=lighteval_g1h_policy.SamplingMethod.GENERATIVE,
            corpus_level_fn=lambda values: sum(values) / len(values),
            higher_is_better=True,
        )
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=8, name="avg@8")

        self.assertEqual(wrapped.metric_name, Metrics.exact_match.value.metric_name)
        self.assertIsInstance(
            wrapped.sample_level_fn,
            type(Metrics.exact_match.value.sample_level_fn),
        )
        self.assertNotIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)

    def test_avg_wrapper_preserves_official_grouped_metric(self) -> None:
        from lighteval.tasks.registry import Registry

        metric = Registry.load_all_task_configs(custom_tasks=None)["ifeval"].metrics[0]
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=8, name="avg@8")

        self.assertIsNot(wrapped, metric)
        self.assertEqual(wrapped.metric_name, metric.metric_name)
        self.assertEqual(
            type(wrapped.sample_level_fn),
            type(metric.sample_level_fn),
        )

    def test_avg_wrapper_preserves_native_lighteval_judge(self) -> None:
        from lighteval.metrics.metrics_sample import JudgeLLM

        metric = SimpleNamespace(
            sample_level_fn=object.__new__(JudgeLLM),
            metric_name=["declared_name"],
            corpus_level_fn={"different_output_name": lambda values: sum(values) / len(values)},
            higher_is_better={"different_output_name": True},
        )
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=8, name="avg@8_0")

        self.assertIsInstance(wrapped, SimpleNamespace)
        self.assertEqual(wrapped.metric_name, ["declared_name"])
        self.assertIsInstance(wrapped.sample_level_fn, JudgeLLM)

    def test_avg_wrapper_converts_native_pass_at_k_to_real_avg(self) -> None:
        from lighteval.metrics.metrics_sample import PassAtK

        metric = SimpleNamespace(
            metric_name="pass@1",
            sample_level_fn=PassAtK(n=8, k=[1]),
            category=lighteval_g1h_policy.SamplingMethod.GENERATIVE,
            corpus_level_fn=lambda values: sum(values) / len(values),
            higher_is_better=True,
        )
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=8, name="avg@8")
        self.assertIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(wrapped.sample_level_fn.n, 8)
        self.assertEqual(type(wrapped.sample_level_fn.compute_score.__self__).__name__, "_SinglePredictionScorer")

    def test_avg_wrapper_converts_gpass_to_real_avg(self) -> None:
        from lighteval.metrics.metrics import Metrics

        wrapped = lighteval_g1h_policy._avg_metric(
            Metrics.g_pass_at_k.value,
            k=3,
            name="avg@3",
        )
        score = wrapped.sample_level_fn.compute(
            Doc(
                task_name="toy",
                query="Pick one.",
                choices=["A", "B"],
                gold_index=0,
            ),
            ModelResponse(text=["A", "B", "A"]),
        )

        self.assertIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(wrapped.sample_level_fn.n, 3)
        self.assertEqual(score, 2 / 3)

    def test_avg_policy_prefers_native_avg_over_duplicate_pass_metric(self) -> None:
        from lighteval.metrics.metrics_sample import AvgAtN, PassAtK

        avg_metric = SimpleNamespace(sample_level_fn=AvgAtN())
        pass_metric = SimpleNamespace(sample_level_fn=PassAtK())

        selected = lighteval_g1h_policy._metrics_for_avg(
            [pass_metric, avg_metric]
        )

        self.assertEqual(selected, [avg_metric])

    def test_aime_policy_emits_one_native_avg_metric(self) -> None:
        from lighteval.tasks.tasks.aime import aime24

        configured = lighteval_g1h_policy._policy_config(
            aime24,
            canonical_name="aime24",
            policy={
                "avg_k": 8,
                "rollout_n": 8,
                "long_rollout_tasks": [],
                "zero_shot": True,
                "generation_size": 6144,
                "gpass_generation_size": 6144,
            },
        )

        self.assertEqual(len(configured.metrics), 1)
        self.assertEqual(configured.metrics[0].metric_name, "avg@8")
        self.assertEqual(configured.num_samples, [8])

    def test_policy_config_prefers_local_mmlu_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "cache/lighteval_mmlu/sociology.jsonl"
            cached.parent.mkdir(parents=True)
            cached.write_text('{"question": "q"}\n', encoding="utf-8")
            source = SimpleNamespace(
                name="mmlu:sociology",
                full_name="mmlu:sociology|0",
                prompt_function=lambda line, task_name=None: None,
                hf_repo="lighteval/mmlu",
                hf_subset="sociology",
                hf_data_files=None,
                hf_avail_splits=("auxiliary_train", "test", "validation", "dev"),
                evaluation_splits=("test",),
                few_shots_split="dev",
                few_shots_select="sequential",
                num_fewshots=5,
                metrics=[
                    SimpleNamespace(
                        metric_name="em",
                        sample_level_fn=lambda doc, model_response: 0.0,
                        higher_is_better=True,
                    )
                ],
                num_samples=[1],
                generation_size=64,
            )
            policy = {
                "avg_k": 8,
                "rollout_n": 8,
                "long_rollout_tasks": [],
                "zero_shot": True,
                "generation_size": 2048,
                "gpass_generation_size": 4096,
            }
            with mock.patch.dict(os.environ, {"DATASETS_PATH": tmp}, clear=False):
                configured = lighteval_g1h_policy._policy_config(
                    source,
                    canonical_name="mmlu:sociology",
                    policy=policy,
                )

        self.assertEqual(configured.hf_repo, "json")
        self.assertEqual(configured.hf_subset, "default")
        self.assertEqual(configured.hf_data_files, {"test": str(cached)})
        self.assertEqual(configured.evaluation_splits, ("test",))
        self.assertIsNone(configured.few_shots_split)

    def test_policy_config_prefers_authorized_local_gpqa_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "gpqa/main.jsonl"
            cached.parent.mkdir(parents=True)
            cached.write_text('{"question": "q"}\n', encoding="utf-8")
            source = SimpleNamespace(
                hf_repo="Idavidrein/gpqa",
                hf_subset="gpqa_main",
                hf_data_files=None,
                hf_avail_splits=("train",),
                evaluation_splits=("train",),
                few_shots_split=None,
                prompt_function=lambda line, task_name=None: None,
            )
            with mock.patch.dict(os.environ, {"DATASETS_PATH": tmp}, clear=False):
                configured = lighteval_g1h_policy._prefer_local_dataset(
                    source,
                    canonical_name="gpqa:mc",
                )

        self.assertEqual(configured.hf_repo, "json")
        self.assertEqual(configured.hf_subset, "default")
        self.assertEqual(configured.hf_data_files, {"train": str(cached)})
        doc = configured.prompt_function(
            {
                "question": "Which?",
                "answer": "C",
                "A": "one",
                "B": "two",
                "C": "three",
                "D": "four",
            },
            "gpqa:mc",
        )
        self.assertEqual(doc.gold_index, 2)
        self.assertEqual(doc.choices, ["A", "B", "C", "D"])
        self.assertIn("C. three", doc.query)

    def test_cached_dataset_file_prefers_existing_datasets_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cached = Path(tmp) / "cache" / "human_eval" / "HumanEval.jsonl.gz"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"fixture")
            with mock.patch.dict(os.environ, {"DATASETS_PATH": tmp}, clear=False):
                selected = lighteval_rwkv_skills_tasks._cached_dataset_file(
                    "cache/human_eval/HumanEval.jsonl.gz",
                    "https://example.invalid/HumanEval.jsonl.gz",
                )

        self.assertEqual(selected, str(cached))

    def test_avg_wrapper_preserves_native_scorer_for_generative_mcq(self) -> None:
        from lighteval.metrics.metrics import Metrics

        metric = Metrics.gpqa_instruct_metric.value
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=2, name="avg@2")
        doc = Doc(
            task_name="gpqa:main",
            query="Pick one.\nA. first\nB. second\nC. third\nD. fourth",
            gold_index=2,
            choices=["A", "B", "C", "D"],
            specific={},
        )

        native_score = metric.sample_level_fn.compute(
            doc,
            ModelResponse(text=["The correct answer is C."]),
        )
        wrapped_score = wrapped.sample_level_fn.compute(
            doc,
            ModelResponse(text=["The correct answer is C.", "The correct answer is A."]),
        )

        self.assertIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(wrapped_score, native_score / 2)

    def test_logprob_metric_gets_a_real_avg_generation_branch(self) -> None:
        from lighteval.metrics.metrics import Metrics

        metric = Metrics.loglikelihood_acc.value
        wrapped = lighteval_g1h_policy._avg_metric(metric, k=8, name="avg@8")
        self.assertEqual(wrapped.category, lighteval_g1h_policy.SamplingMethod.GENERATIVE)
        self.assertIsInstance(wrapped.sample_level_fn, lighteval_g1h_policy.AvgAtN)
        self.assertEqual(wrapped.sample_level_fn.n, 8)

    def test_logprob_avg_policy_drives_real_rollout_count(self) -> None:
        from lighteval.tasks.registry import Registry

        source = Registry.load_all_task_configs(custom_tasks=None, load_multilingual=True)[
            "ceval_zho_mcf:college_programming"
        ]
        configured = lighteval_g1h_policy._policy_config(
            source,
            canonical_name="ceval_zho_mcf:college_programming",
            policy={
                "metric": "avg",
                "avg_k": 3,
                "rollout_n": 3,
                "long_rollout_tasks": [],
                "zero_shot": True,
                "generation_size": 256,
                "gpass_generation_size": 256,
            },
        )
        self.assertEqual(configured.metrics[0].metric_name, "avg@3")
        self.assertEqual(configured.metrics[0].category, lighteval_g1h_policy.SamplingMethod.GENERATIVE)
        self.assertEqual(configured.metrics[0].sample_level_fn.n, 3)
        self.assertEqual(configured.num_samples, [3])

    def test_native_policy_keeps_logprob_metric_unchanged(self) -> None:
        from lighteval.tasks.registry import Registry

        source = Registry.load_all_task_configs(custom_tasks=None, load_multilingual=True)[
            "ceval_zho_mcf:college_programming"
        ]
        configured = lighteval_g1h_policy._policy_config(
            source,
            canonical_name="ceval_zho_mcf:college_programming",
            policy={
                "metric": "native",
                "avg_k": 8,
                "rollout_n": 8,
                "long_rollout_tasks": [],
                "zero_shot": True,
                "generation_size": 256,
                "gpass_generation_size": 256,
            },
        )
        self.assertEqual(len(configured.metrics), 1)
        self.assertEqual(configured.metrics[0].metric_name, "acc")
        self.assertEqual(type(configured.metrics[0].sample_level_fn).__name__, "LoglikelihoodAcc")
        self.assertEqual(configured.metrics[0].category, lighteval_g1h_policy.SamplingMethod.LOGPROBS)















    def test_lighteval_tasks_loads_rwkv_skills_registry_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "benchmark_registry.py"
            source.write_text(
                "\n".join(
                    [
                        "_EXPLICIT_METADATA: dict[str, object] = {",
                        '    canonical_slug("gsm8k"): _math("gsm8k"),',
                        '    canonical_slug("mmlu"): _knowledge("mmlu"),',
                        '    canonical_slug("bfcl_v3"): _function_calling("bfcl_v3", scheduler_jobs=()),',
                        "}",
                    ]
                )
            )

            rows = lighteval_tasks.load_source_benchmarks(str(source), "auto")

        self.assertEqual(
            rows,
            [
                lighteval_tasks.SourceBenchmark(name="gsm8k", field="maths"),
                lighteval_tasks.SourceBenchmark(name="mmlu", field="knowledge"),
                lighteval_tasks.SourceBenchmark(name="bfcl_v3", field="function_calling"),
            ],
        )

    def test_lighteval_export_plan_exports_details(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_lighteval_export_plan(
            lighteval_export_args(output="tmp/gsm8k.jsonl"),
            root=ROOT,
            env={},
            config=loaded_config,
        )

        self.assertEqual(plan.command[1:3], ["-m", "helicopter_cli.lighteval_export"])
        self.assertIn(str(ROOT / "results/lighteval/details/run"), plan.command)
        options = command_options(plan.command)
        self.assertEqual(options["--output"], "tmp/gsm8k.jsonl")
        self.assertEqual(options["--format"], "jsonl")

    def test_takeoff_plan_uses_verl_module_entrypoint_and_default_overrides(self) -> None:
        loaded_config = load_example_config()
        venv_python = ROOT / ".venv/bin/python"

        plan = build_takeoff_plan(loaded_config, venv_python=venv_python)
        overrides = hydra_map(plan)
        optional_rollout_keys = {
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
        }

        self.assertEqual(plan.cwd, ROOT / "src/train/verl-rwkv")
        self.assertEqual(
            plan.command[:3],
            [
                str(venv_python),
                "-m",
                "verl.experimental.one_step_off_policy.main_ppo",
            ],
        )
        self.assertEqual(
            plan.shown_env,
            {
                "PYTHON": str(venv_python),
                "PYTHONPATH": str((ROOT / "../vllm-rwkv").resolve()),
                "RWKV_LM_PATH": str(ROOT / "src/train/rwkv-lm"),
                "RWKV_MODEL_PATH": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "VLLM_RWKV7_EMB_DEVICE": "gpu",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
            },
        )
        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "actor_rollout_ref.actor.use_dynamic_bsz",
                    "actor_rollout_ref.model.path",
                    "actor_rollout_ref.rollout.name",
                    "actor_rollout_ref.hybrid_engine",
                    "trainer.total_epochs",
                )
            },
            {
                "actor_rollout_ref.actor.use_dynamic_bsz": "False",
                "actor_rollout_ref.model.path": "/weights/RWKV/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                "actor_rollout_ref.rollout.name": "vllm",
                "actor_rollout_ref.hybrid_engine": "False",
                "trainer.total_epochs": "2",
            },
        )
        self.assertEqual(optional_rollout_keys & overrides.keys(), set())

    def test_takeoff_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()
        plan = build_takeoff_plan(
            loaded_config,
            loaded_env={
                "WEIGHT_PATH": "/weights/RWKV",
                "DATASETS_PATH": "/datasets",
                "HELICOPTER_VLLM_RWKV_PATH": "../vllm-rwkv",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
                "VLLM_MAX_NUM_BATCHED_TOKENS": "65536",
                "VLLM_RWKV_PATH": "legacy/path",
                "VLLM_RWKV7_EMB_DEVICE": "cpu",
                "VLLM_USE_V2_MODEL_RUNNER": "1",
            },
        )
        overrides = hydra_map(plan)
        forbidden_env_keys = {
            "VLLM_GPU_MEMORY_UTILIZATION",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            "VLLM_RWKV_PATH",
            "VLLM_USE_V2_MODEL_RUNNER",
        }
        forbidden_override_keys = {
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.max_num_seqs",
            "actor_rollout_ref.rollout.max_num_batched_tokens",
        }

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertEqual(plan.env["VLLM_RWKV7_EMB_DEVICE"], "gpu")
        self.assertEqual(plan.env["PYTHONPATH"], str((ROOT / "../vllm-rwkv").resolve()))
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_override_keys & overrides.keys(), set())

    def test_infer_runtime_env_strips_dotenv_vllm_knobs(self) -> None:
        loaded_config = load_example_config()

        plan = commands.build_infer_plan(
            infer_args(),
            root=ROOT,
            env={
                "WEIGHT_PATH": "/weights/RWKV",
                "VLLM_RWKV7_WKV_MODE": "fp32io16",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.85",
                "VLLM_MAX_NUM_SEQS": "2048",
            },
            config=loaded_config,
        )
        options = command_options(plan.command)
        forbidden_env_keys = {"VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_NUM_SEQS"}
        forbidden_option_keys = {"--gpu-memory-utilization", "--max-num-seqs"}

        self.assertEqual(plan.env["VLLM_RWKV7_WKV_MODE"], "fp32io16")
        self.assertEqual(forbidden_env_keys & plan.env.keys(), set())
        self.assertEqual(forbidden_option_keys & options.keys(), set())

    def test_takeoff_config_adv_estimator_becomes_hydra_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {**takeoff["grpo"], "adv_estimator": "maxrl", "reward_manager": "dapo"}

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                "algorithm.adv_estimator": overrides["algorithm.adv_estimator"],
                "reward.reward_manager.name": overrides["reward.reward_manager.name"],
            },
            {
                "algorithm.adv_estimator": "maxrl",
                "reward.reward_manager.name": "dapo",
            },
        )

    def test_takeoff_rollout_gpu_count_becomes_top_level_and_actor_rollout_overrides(self) -> None:
        loaded_config = load_example_config()
        takeoff = loaded_config["takeoff"]
        takeoff["grpo"] = {
            **takeoff["grpo"],
            "trainer_n_gpus_per_node": 7,
            "rollout_n_gpus_per_node": 1,
            "rollout_data_parallel_size": 1,
            "rollout_pipeline_parallel_size": 1,
        }

        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                key: overrides[key]
                for key in (
                    "trainer.n_gpus_per_node",
                    "rollout.n_gpus_per_node",
                    "actor_rollout_ref.rollout.n_gpus_per_node",
                    "actor_rollout_ref.rollout.data_parallel_size",
                    "actor_rollout_ref.rollout.pipeline_model_parallel_size",
                )
            },
            {
                "trainer.n_gpus_per_node": "7",
                "rollout.n_gpus_per_node": "1",
                "actor_rollout_ref.rollout.n_gpus_per_node": "1",
                "actor_rollout_ref.rollout.data_parallel_size": "1",
                "actor_rollout_ref.rollout.pipeline_model_parallel_size": "1",
            },
        )

    def test_takeoff_dataset_files_become_verl_file_lists(self) -> None:
        loaded_config = load_example_config()
        datasets = loaded_config["datasets"]
        datasets["dapo_math_17k"] = {
            "train_files": ["${DATASETS_PATH}/DAPO/dapo-math-17k.parquet"],
            "val_files": [
                "${DATASETS_PATH}/AIME24/test.parquet",
                "${DATASETS_PATH}/AIME25/test.parquet",
            ],
        }

        plan = build_takeoff_plan(loaded_config, args=takeoff_args(dataset="dapo_math_17k"))
        overrides = hydra_map(plan)

        self.assertEqual(
            {
                "data.train_files": overrides["data.train_files"],
                "data.val_files": overrides["data.val_files"],
            },
            {
                "data.train_files": "['/datasets/DAPO/dapo-math-17k.parquet']",
                "data.val_files": "['/datasets/AIME24/test.parquet','/datasets/AIME25/test.parquet']",
            },
        )
        self.assertEqual(
            set(plan.shown_env),
            {
                "PYTHON",
                "PYTHONPATH",
                "RWKV_LM_PATH",
                "RWKV_MODEL_PATH",
                "VLLM_RWKV7_EMB_DEVICE",
                "VLLM_RWKV7_WKV_MODE",
            },
        )

    def test_takeoff_defaults_keep_actor_kl_loss_disabled(self) -> None:
        loaded_config = load_example_config()
        overrides = hydra_map(build_takeoff_plan(loaded_config))

        self.assertEqual(
            {
                "actor_rollout_ref.actor.use_kl_loss": overrides["actor_rollout_ref.actor.use_kl_loss"],
                "actor_rollout_ref.actor.kl_loss_coef": overrides["actor_rollout_ref.actor.kl_loss_coef"],
            },
            {
                "actor_rollout_ref.actor.use_kl_loss": "False",
                "actor_rollout_ref.actor.kl_loss_coef": "0.0",
            },
        )

    def test_takeoff_explicit_dataset_files_do_not_require_dataset_root(self) -> None:
        loaded_config = load_example_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.pth"
            model_path.write_bytes(b"")
            external_vllm = tmp_path / "vllm-rwkv"
            external_vllm.mkdir()
            loaded_config["paths"]["vllm_rwkv_path"] = str(external_vllm)
            missing_dataset_root = tmp_path / "missing-datasets"
            loaded_config["models"]["local-test"] = {
                "path": str(model_path),
                "served_model_name": "local-test",
                "max_model_len": 8192,
            }
            loaded_config["datasets"]["dapo_math_17k"] = {
                "train_files": ["${DATASETS_PATH}/DAPO/dapo-math-17k.parquet"],
                "val_files": ["${DATASETS_PATH}/AIME24/test.parquet"],
            }
            args = Namespace(
                algorithm="grpo",
                model="local-test",
                dataset="dapo_math_17k",
                dry_run=False,
                wkv_mode=None,
                emb_device=None,
                num_nodes=None,
                num_devices=None,
                override=None,
            )

            plan = commands.build_takeoff_plan(
                args,
                root=ROOT,
                env={
                    "DATASETS_PATH": str(missing_dataset_root),
                    "HELICOPTER_PYTHON": "/usr/bin/python3",
                },
                config=loaded_config,
            )

        self.assertEqual(
            hydra_map(plan)["data.train_files"],
            f"['{missing_dataset_root}/DAPO/dapo-math-17k.parquet']",
        )

    def test_takeoff_partial_explicit_dataset_files_require_dataset_root(self) -> None:
        loaded_config = load_example_config()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.pth"
            model_path.write_bytes(b"")
            external_vllm = tmp_path / "vllm-rwkv"
            external_vllm.mkdir()
            loaded_config["paths"]["vllm_rwkv_path"] = str(external_vllm)
            missing_dataset_root = tmp_path / "missing-datasets"
            loaded_config["models"]["local-test"] = {
                "path": str(model_path),
                "served_model_name": "local-test",
                "max_model_len": 8192,
            }
            loaded_config["datasets"]["partial"] = {
                "train_files": ["${DATASETS_PATH}/partial/train.parquet"],
            }
            args = Namespace(
                algorithm="grpo",
                model="local-test",
                dataset="partial",
                dry_run=False,
                wkv_mode=None,
                emb_device=None,
                num_nodes=None,
                num_devices=None,
                override=None,
            )

            with self.assertRaises(SystemExit) as raised:
                commands.build_takeoff_plan(
                    args,
                    root=ROOT,
                    env={
                        "DATASETS_PATH": str(missing_dataset_root),
                        "HELICOPTER_PYTHON": "/usr/bin/python3",
                    },
                    config=loaded_config,
                )

        self.assertEqual(
            str(raised.exception),
            f"dataset root not found: {missing_dataset_root / 'partial'}",
        )

    def test_takeoff_user_overrides_are_appended_after_generated_overrides(self) -> None:
        loaded_config = load_example_config()
        plan = build_takeoff_plan(
            loaded_config,
            args=takeoff_args(override=["trainer.total_epochs=1", "trainer.save_freq=10"]),
        )

        self.assertEqual(hydra_values(plan, "trainer.total_epochs"), ["2", "1"])
        self.assertEqual(hydra_values(plan, "trainer.save_freq"), ["20", "10"])
        self.assertEqual(plan.command[-2:], ["trainer.total_epochs=1", "trainer.save_freq=10"])

    def test_takeoff_rejects_missing_default_venv_python(self) -> None:
        config = load_example_config()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"HELICOPTER_PYTHON", "PYTHON", "HELICOPTER_VENV", "VENV", "REMOTE_VENV"}
        }
        venv_python = ROOT / ".venv/bin/python"
        original_exists = Path.exists

        with mock.patch.object(Path, "exists", autospec=True) as exists:
            exists.side_effect = lambda path: False if path == venv_python else original_exists(path)
            with self.assertRaises(SystemExit) as raised:
                commands.python_executable(config, root=ROOT, env=env, require_configured=True)

        self.assertEqual(
            str(raised.exception),
            f"Python executable not found: {venv_python}; run scripts/install_local.sh "
            "or set HELICOPTER_PYTHON / paths.python",
        )


def eval_run_args(**overrides: object) -> Namespace:
    values = {
        **vars(lighteval_args()),
        "tasks": None,
        "dry_run": True,
        "wkv_mode": None,
        "emb_device": None,
        "tensor_parallel_size": None,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
        "vllm_env": None,
        "no_server": False,
        "keep_server": False,
        "server_timeout": eval_run.DEFAULT_SERVER_TIMEOUT_S,
        "scoreboard": False,
    }
    values.update(overrides)
    return Namespace(**values)


class EvalRunTests(unittest.TestCase):
    def test_pinned_scoreboard_task_id_only_applies_to_score_stage(self) -> None:
        self.assertEqual(
            eval_run._pinned_scoreboard_task_id(
                {
                    "HELICOPTER_PIPELINE_STAGE": "score",
                    "HELICOPTER_SCOREBOARD_TASK_ID": "70",
                }
            ),
            "70",
        )
        self.assertIsNone(
            eval_run._pinned_scoreboard_task_id(
                {
                    "HELICOPTER_PIPELINE_STAGE": "generate",
                    "HELICOPTER_SCOREBOARD_TASK_ID": "70",
                }
            )
        )
        self.assertIsNone(
            eval_run._pinned_scoreboard_task_id({"HELICOPTER_PIPELINE_STAGE": "score"})
        )

    def test_resolve_run_tasks_prefers_cli_then_config(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": ["gsm8k|0", "mmlu|0"]}
        self.assertEqual(
            eval_run.resolve_run_tasks(eval_run_args(tasks="aime24|0"), loaded),
            "aime24|0",
        )
        self.assertEqual(
            eval_run.resolve_run_tasks(eval_run_args(), loaded),
            "gsm8k|0,mmlu|0",
        )

    def test_resolve_run_tasks_requires_tasks(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {key: value for key, value in loaded.get("lighteval", {}).items() if key != "tasks"}
        with self.assertRaises(SystemExit):
            eval_run.resolve_run_tasks(eval_run_args(), loaded)

    def test_port_and_health_url_helpers(self) -> None:
        self.assertEqual(eval_run.port_from_base_url("http://127.0.0.1:8000/v1"), "8000")
        self.assertIsNone(eval_run.port_from_base_url("http://example.com/v1"))
        self.assertEqual(
            eval_run.health_url_for("http://127.0.0.1:8000/v1"),
            "http://127.0.0.1:8000/v1/models",
        )

    def test_dry_run_prints_infer_and_lighteval_plans(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": "gsm8k|0"}
        args = eval_run_args()
        env = {"WEIGHT_PATH": "/weights/RWKV"}
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(args, root=ROOT, env=env, config=loaded)
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("vllm serve", lines[0])
        self.assertIn("--port 8000", lines[0])
        self.assertIn("-m lighteval endpoint litellm", lines[1])
        self.assertIn("gsm8k|0", lines[1])

    def test_dry_run_with_remote_base_url_skips_server_plan(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {
            **loaded.get("lighteval", {}),
            "tasks": "gsm8k|0",
            "base_url": "http://eval-farm.internal:9000/v1",
        }
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(
                eval_run_args(), root=ROOT, env={"WEIGHT_PATH": "/weights/RWKV"}, config=loaded
            )
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn("-m lighteval endpoint litellm", lines[0])
        self.assertNotIn("vllm serve", lines[0])

    def test_no_server_flag_skips_server_plan(self) -> None:
        loaded = load_example_config()
        loaded["lighteval"] = {**loaded.get("lighteval", {}), "tasks": "gsm8k|0"}
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_run.run_eval(
                eval_run_args(no_server=True),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=loaded,
            )
        self.assertEqual(exit_code, 0)
        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertNotIn("vllm serve", lines[0])

    def test_infer_args_namespace_forwards_server_overrides(self) -> None:
        args = eval_run_args(
            wkv_mode="fp16",
            gpu_memory_utilization=0.5,
            vllm_env=["VLLM_WSL2_ENABLE_PIN_MEMORY=1"],
        )
        namespace = eval_run.infer_args_namespace(args, port="8123")
        self.assertEqual(namespace.model, "g1g-1.5b")
        self.assertEqual(namespace.port, "8123")
        self.assertEqual(namespace.wkv_mode, "fp16")
        self.assertEqual(namespace.gpu_memory_utilization, 0.5)
        self.assertEqual(namespace.vllm_env, ["VLLM_WSL2_ENABLE_PIN_MEMORY=1"])
        self.assertIsNone(namespace.host)

    def test_scoreboard_dataset_name_strips_fewshot_and_suite(self) -> None:
        self.assertEqual(eval_run.scoreboard_dataset_name("gsm8k|0"), "gsm8k")
        self.assertEqual(eval_run.scoreboard_dataset_name("extended|mmlu|5"), "mmlu")
        self.assertEqual(eval_run.scoreboard_dataset_name("apibank_level1"), "apibank_level1")

    def test_scoreboard_model_name_prefers_served_model_name(self) -> None:
        loaded = load_example_config()
        self.assertEqual(eval_run.scoreboard_model_name(eval_run_args(), loaded), "g1g-1.5b")
        self.assertEqual(
            eval_run.scoreboard_model_name(
                eval_run_args(lighteval_model_name="custom-name"), loaded
            ),
            "custom-name",
        )


def batch_args(**overrides: object) -> Namespace:
    values = {
        **vars(eval_run_args()),
        "models": None,
        "tasks": None,
        "tasks_from_db": False,
        "benchmark_scope": None,
        "benchmark_fields": None,
        "benchmark_limit": None,
        "fc_tasks": None,
        "gpus": None,
        "gpu_idle_max_mem": None,
        "parallel": None,
        "max_retries": 0,
        "port_base": None,
        "batch_output": None,
        "rerun": False,
    }
    values.update(overrides)
    return Namespace(**values)


class EvalBatchTests(unittest.TestCase):
    def test_resolve_batch_plan_accepts_explicit_model_benchmark_jobs(self) -> None:
        loaded = load_example_config()
        units = eval_batch.resolve_batch_plan(
            batch_args(
                jobs=[
                    "g1d-0.4b=gsm8k|0,mmlu|0",
                    "g1g-1.5b=fc:bfcl_v3",
                ]
            ),
            loaded,
        )
        self.assertEqual(
            [(unit.model, unit.kind, unit.tasks) for unit in units],
            [
                ("g1d-0.4b", "lighteval", ["gsm8k|0"]),
                ("g1d-0.4b", "lighteval", ["mmlu|0"]),
                ("g1g-1.5b", "fc", ["bfcl_v3"]),
            ],
        )

    def test_explicit_jobs_cannot_be_mixed_with_matrix_selection(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cannot be combined"):
            eval_batch.resolve_batch_plan(
                batch_args(jobs=["g1d-0.4b=gsm8k|0"], models=["g1d-0.4b"]),
                load_example_config(),
            )

    def test_resolve_batch_plan_from_args_and_config(self) -> None:
        loaded = load_example_config()
        loaded["eval"] = {"batch": {"models": ["g1d-0.4b"], "tasks": ["gsm8k|0"], "fc_tasks": ["bfcl_v3"]}}
        units = eval_batch.resolve_batch_plan(batch_args(), loaded)
        self.assertEqual(
            [(unit.model, unit.kind, unit.tasks) for unit in units],
            [("g1d-0.4b", "lighteval", ["gsm8k|0"]), ("g1d-0.4b", "fc", ["bfcl_v3"])],
        )

        units = eval_batch.resolve_batch_plan(
            batch_args(models=["a,b"], tasks=["gsm8k|0,mmlu|0"], fc_tasks=None),
            loaded,
        )
        self.assertEqual(
            [(unit.model, unit.kind) for unit in units],
            [
                ("a", "lighteval"),
                ("b", "lighteval"),
                ("a", "lighteval"),
                ("b", "lighteval"),
            ],
        )
        self.assertEqual(units[0].tasks, ["gsm8k|0"])
        self.assertEqual(units[1].tasks, ["gsm8k|0"])
        self.assertEqual(units[2].tasks, ["mmlu|0"])

    def test_catalog_rows_are_interleaved_by_field(self) -> None:
        rows = [
            {"field": "math", "name": "math-1"},
            {"field": "math", "name": "math-2"},
            {"field": "coding", "name": "coding-1"},
            {"field": "coding", "name": "coding-2"},
            {"field": "instruction_following", "name": "ifeval-1"},
            {"field": "knowledge", "name": "knowledge-1"},
        ]

        interleaved = eval_batch.interleave_catalog_rows(rows)

        self.assertEqual(
            [row["name"] for row in interleaved],
            ["math-1", "coding-1", "ifeval-1", "knowledge-1", "math-2", "coding-2"],
        )

    def test_resolve_batch_plan_requires_models_and_benchmarks(self) -> None:
        loaded = load_example_config()
        with self.assertRaises(SystemExit):
            eval_batch.resolve_batch_plan(batch_args(), loaded)
        with self.assertRaises(SystemExit):
            eval_batch.resolve_batch_plan(batch_args(models=["a"]), loaded)

    def test_resolve_slots_explicit_gpus_and_port_base(self) -> None:
        loaded = load_example_config()
        slots = eval_batch.resolve_slots(batch_args(gpus="1,3", port_base=9000), loaded, {})
        self.assertEqual([(slot.gpu, slot.port) for slot in slots], [(1, 9000), (3, 9001)])

    def test_resolve_slots_external_endpoint_is_single_generic_slot(self) -> None:
        loaded = load_example_config()
        for kwargs in ({"no_server": True}, {"base_url": "http://farm:9000/v1"}):
            slots = eval_batch.resolve_slots(batch_args(**kwargs), loaded, {})
            self.assertEqual(len(slots), 1)
            self.assertIsNone(slots[0].gpu)

        slots = eval_batch.resolve_slots(batch_args(no_server=True, parallel=3), loaded, {})
        self.assertEqual(len(slots), 1)
        self.assertIsNone(slots[0].gpu)

    def test_parallel_is_only_an_optional_cap(self) -> None:
        loaded = load_example_config()
        self.assertIsNone(eval_batch.resolve_parallel_cap(batch_args(), loaded))
        loaded["eval"] = {"batch": {"parallel": 4}}
        self.assertEqual(eval_batch.resolve_parallel_cap(batch_args(), loaded), 4)
        self.assertEqual(eval_batch.resolve_parallel_cap(batch_args(parallel=2), loaded), 2)

    def test_postprocess_workers_follow_scheduler_score_capacity(self) -> None:
        self.assertEqual(
            eval_batch.derive_postprocess_workers(runnable_count=120, score_capacity=10),
            10,
        )
        self.assertEqual(
            eval_batch.derive_postprocess_workers(runnable_count=3, score_capacity=10),
            3,
        )
        self.assertEqual(
            eval_batch.derive_postprocess_workers(
                runnable_count=120, score_capacity=10, configured_ceiling=6
            ),
            6,
        )

    def test_two_endpoint_replicas_allow_two_benchmark_owners(self) -> None:
        loaded = {
            "models": {
                "rwkv": {
                    "served_model_name": "rwkv-served",
                    "replicas": [
                        {"name": "gpu-a", "base_url": "http://a:8000/v1", "max_num_seqs": 64},
                        {"name": "gpu-b", "base_url": "http://b:8000/v1", "max_num_seqs": 64},
                    ],
                }
            }
        }
        replicas = eval_batch.resolve_model_replicas(loaded, "rwkv")
        self.assertEqual([replica.name for replica in replicas], ["gpu-a", "gpu-b"])
        plan = eval_batch.derive_model_concurrency(
            model="rwkv",
            pending_benchmarks=30,
            rollout_n=8,
            max_num_seqs=64,
            configured_request_ceiling=8,
            replica_count=len(replicas),
        )
        self.assertEqual(plan.benchmark_workers, 2)
        self.assertEqual(plan.concurrent_requests, 8)

    def test_replica_overrides_only_endpoint_runtime_not_logical_model(self) -> None:
        unit = eval_batch.BatchUnit(model="rwkv", kind="lighteval", tasks=["math_500|0"])
        replica = eval_batch.ModelReplica(
            name="gpu-b",
            base_url="http://b:8000/v1",
            served_model_name="rwkv-served",
            api_key_env="MODEL_API_KEY",
        )
        unit_args = eval_batch._unit_args(
            batch_args(),
            unit,
            eval_batch.GpuSlot(index=0, gpu=None, port=0),
            replica=replica,
            env={"MODEL_API_KEY": "secret"},
        )
        self.assertEqual(unit_args.model, "rwkv")
        self.assertEqual(unit_args.base_url, "http://b:8000/v1")
        self.assertEqual(unit_args.lighteval_model_name, "rwkv-served")
        self.assertEqual(unit_args.api_key, "secret")
        self.assertTrue(unit_args.no_server)

    def test_model_concurrency_is_derived_from_model_capacity_and_rollout_k(self) -> None:
        plan = eval_batch.derive_model_concurrency(
            model="rwkv",
            pending_benchmarks=6,
            rollout_n=8,
            max_num_seqs=64,
            configured_request_ceiling=8,
        )
        self.assertEqual(plan.benchmark_workers, 1)
        self.assertEqual(plan.concurrent_requests, 8)
        self.assertEqual(plan.benchmark_workers * plan.concurrent_requests * plan.rollout_n, 64)

        backpressured = eval_batch.derive_model_concurrency(
            model="rwkv",
            pending_benchmarks=6,
            rollout_n=8,
            max_num_seqs=64,
            configured_request_ceiling=8,
            waiting_requests=1,
            source="server_info",
        )
        self.assertEqual(backpressured.benchmark_workers, 1)
        self.assertEqual(backpressured.concurrent_requests, 1)

    def test_model_concurrency_reserves_process_file_descriptors(self) -> None:
        self.assertEqual(eval_batch.request_cap_from_open_files(1024), 224)
        self.assertIsNone(eval_batch.request_cap_from_open_files(None))

        plan = eval_batch.derive_model_concurrency(
            model="rwkv",
            pending_benchmarks=12,
            rollout_n=1,
            max_num_seqs=1024,
            configured_request_ceiling=None,
            source="model_config",
            open_file_limit=1024,
        )
        self.assertEqual(plan.benchmark_workers, 1)
        self.assertEqual(plan.concurrent_requests, 224)
        self.assertEqual(plan.source, "model_config+nofile:1024")

    def test_model_concurrency_keeps_configured_ceiling_without_server_info(self) -> None:
        plan = eval_batch.derive_model_concurrency(
            model="rwkv",
            pending_benchmarks=12,
            rollout_n=4,
            max_num_seqs=None,
            configured_request_ceiling=8,
        )
        self.assertEqual(plan.concurrent_requests, 8)

    def test_unit_args_assigns_slot_base_url_and_joined_tasks(self) -> None:
        unit = eval_batch.BatchUnit(model="m", kind="lighteval", tasks=["gsm8k|0", "mmlu|0"])
        slot = eval_batch.GpuSlot(index=1, gpu=3, port=8001)
        args = batch_args()
        args._batch_run_dir = "tmp/batch-run"
        unit_args = eval_batch._unit_args(args, unit, slot)
        self.assertEqual(unit_args.model, "m")
        self.assertEqual(unit_args.tasks, "gsm8k|0,mmlu|0")
        self.assertEqual(unit_args.base_url, "http://127.0.0.1:8001/v1")
        self.assertIn("tmp/batch-run", unit_args.output_dir)
        self.assertIn("/lighteval", unit_args.output_dir)
        self.assertFalse(unit_args.keep_server)
        env = eval_batch._unit_env({"PATH": "/bin"}, slot)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_filter_completed_units_drops_scored_tasks(self) -> None:
        loaded = load_example_config()
        units = [
            eval_batch.BatchUnit(model="g1d-0.4b", kind="lighteval", tasks=["gsm8k|0", "mmlu|0"]),
            eval_batch.BatchUnit(model="g1d-0.4b", kind="fc", tasks=["bfcl_v3"]),
        ]

        captured_identities = {}

        async def fake_query(*, model_name, datasets, root, identities=None):
            captured_identities.update(identities or {})
            return {"gsm8k", "bfcl_v3"} & set(datasets)

        with mock.patch.object(eval_batch, "_query_completed_datasets", fake_query):
            eval_batch.filter_completed_units(
                units, args=batch_args(), config=loaded, env={}, root=ROOT
            )
        self.assertEqual(units[0].tasks, ["mmlu|0"])
        self.assertEqual(units[0].skipped_tasks, ["gsm8k|0"])
        self.assertEqual(units[0].status, "pending")
        self.assertEqual(units[1].tasks, [])
        self.assertEqual(units[1].status, "skipped")
        gsm8k_config_path, gsm8k_sampling = captured_identities["gsm8k"]
        self.assertTrue(gsm8k_config_path.endswith("configs/example.toml"))
        self.assertIsInstance(gsm8k_sampling, dict)

    def test_run_unit_retries_and_records_failure(self) -> None:
        loaded = load_example_config()
        unit = eval_batch.BatchUnit(model="g1d-0.4b", kind="lighteval", tasks=["gsm8k|0"])
        slot = eval_batch.GpuSlot(index=0, gpu=None, port=8000)
        calls = []

        def failing_runner(args, *, root, env, config):
            calls.append(args.tasks)
            raise SystemExit("server never became healthy")

        with mock.patch.object(eval_batch, "run_eval", failing_runner):
            eval_batch.run_unit(
                unit,
                args=batch_args(),
                slot=slot,
                root=ROOT,
                env={},
                config=loaded,
                max_retries=1,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(unit.status, "failed")
        self.assertEqual(unit.attempts, 2)
        self.assertIn("healthy", unit.message)

    def test_next_benchmark_waits_for_previous_score_for_same_model(self) -> None:
        loaded = load_example_config()
        score_started = threading.Event()
        allow_score_finish = threading.Event()
        score_finished = threading.Event()
        overlap_observed: list[bool] = []

        def staged_runner(args, *, root, env, config):
            stage = env.get("HELICOPTER_PIPELINE_STAGE")
            if args.tasks == "gsm8k|0" and stage == "score":
                score_started.set()
                allow_score_finish.wait(timeout=2)
                score_finished.set()
            elif args.tasks == "mmlu|0" and stage == "generate":
                score_started.wait(timeout=2)
                overlap_observed.append(score_started.is_set() and not score_finished.is_set())
                allow_score_finish.set()
            return 0

        with mock.patch.object(eval_batch, "run_eval", side_effect=staged_runner):
            exit_code = eval_batch.run_batch(
                batch_args(
                    models=["g1d-0.4b"],
                    tasks=["gsm8k|0,mmlu|0"],
                    no_server=True,
                    dry_run=False,
                    scoreboard=False,
                ),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=loaded,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(overlap_observed, [False])
        self.assertTrue(score_finished.is_set())

    def test_benchmark_barrier_prevents_cross_benchmark_overlap(self) -> None:
        loaded = load_example_config()
        second_model_second_task_started = threading.Event()
        overlap_observed: list[bool] = []

        def staged_runner(args, *, root, env, config):
            if env.get("HELICOPTER_PIPELINE_STAGE") != "generate":
                return 0
            if args.model == "g1d-0.4b" and args.tasks == "gsm8k|0":
                overlap_observed.append(
                    second_model_second_task_started.wait(timeout=2)
                )
            elif args.model == "g1g-1.5b" and args.tasks == "mmlu|0":
                second_model_second_task_started.set()
            return 0

        with mock.patch.object(
            eval_batch,
            "probe_model_runtime",
            return_value=(64, 0, 0, "test"),
        ):
            with mock.patch.object(
                eval_batch,
                "run_eval",
                side_effect=staged_runner,
            ):
                exit_code = eval_batch.run_batch(
                    batch_args(
                        models=["g1d-0.4b,g1g-1.5b"],
                        tasks=["gsm8k|0,mmlu|0"],
                        no_server=True,
                        dry_run=False,
                        scoreboard=False,
                    ),
                    root=ROOT,
                    env={"WEIGHT_PATH": "/weights/RWKV"},
                    config=loaded,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(overlap_observed, [False])

    def test_run_batch_dry_run_prints_plan(self) -> None:
        loaded = load_example_config()
        loaded["eval"] = {"batch": {"models": ["g1d-0.4b"], "tasks": ["gsm8k|0"]}}
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_batch.run_batch(
                batch_args(), root=ROOT, env={"WEIGHT_PATH": "/weights/RWKV"}, config=loaded
            )
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("1 unit(s)", output)
        self.assertIn("vllm serve", output)
        self.assertIn("-m lighteval endpoint litellm", output)

    def test_run_batch_dry_run_returns_failed_exit_for_bad_child_plan(self) -> None:
        loaded = load_example_config()
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = eval_batch.run_batch(
                batch_args(models=["missing-model"], tasks=["gsm8k|0"]),
                root=ROOT,
                env={"WEIGHT_PATH": "/weights/RWKV"},
                config=loaded,
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("missing-model", stdout.getvalue())

    def test_run_batch_loads_lighteval_tasks_from_db_catalog(self) -> None:
        loaded = load_example_config()
        loaded["eval"] = {"batch": {"models": ["g1d-0.4b"]}}
        calls = []

        def successful_runner(args, *, root, env, config):
            calls.append((args.tasks, env.get("HELICOPTER_PIPELINE_STAGE"), args.no_server))
            return 0

        with mock.patch.object(eval_batch, "query_catalog_lighteval_tasks", return_value=["gsm8k", "mmlu:abstract_algebra"]):
            with mock.patch.object(eval_batch, "run_eval", successful_runner):
                exit_code = eval_batch.run_batch(
                    batch_args(tasks_from_db=True, no_server=True, dry_run=False),
                    root=ROOT,
                    env={"WEIGHT_PATH": "/weights/RWKV"},
                    config=loaded,
                )

        self.assertEqual(exit_code, 0)
        self.assertCountEqual(
            calls,
            [
                ("gsm8k", "generate", True),
                ("gsm8k", "score", True),
                ("mmlu:abstract_algebra", "generate", True),
                ("mmlu:abstract_algebra", "score", True),
            ],
        )

    def test_run_batch_isolates_parallel_unit_output_dirs(self) -> None:
        loaded = load_example_config()
        output_dirs = []
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "batch.json"

            def successful_runner(args, *, root, env, config):
                output_dirs.append(args.output_dir)
                return 0

            with mock.patch.object(eval_batch, "run_eval", successful_runner):
                exit_code = eval_batch.run_batch(
                    batch_args(
                        models=["g1d-0.4b,g1g-1.5b"],
                        tasks=["gsm8k|0"],
                        gpus="1,3",
                        parallel=2,
                        dry_run=False,
                        batch_output=str(report_path),
                    ),
                    root=ROOT,
                    env={"WEIGHT_PATH": "/weights/RWKV"},
                    config=loaded,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(output_dirs), 4)
        self.assertEqual(len(set(output_dirs)), 2)
        self.assertTrue(all(str(report_path.with_suffix("")) in path for path in output_dirs))

    def test_run_batch_writes_report_for_real_run(self) -> None:
        loaded = load_example_config()
        loaded["eval"] = {"batch": {"models": ["g1d-0.4b"], "tasks": ["gsm8k|0"]}}
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "batch.json"

            def successful_runner(args, *, root, env, config):
                return 0

            with mock.patch.object(eval_batch, "run_eval", successful_runner):
                exit_code = eval_batch.run_batch(
                    batch_args(
                        dry_run=False,
                        no_server=True,
                        batch_output=str(report_path),
                        max_retries=1,
                        wkv_mode="fp16",
                        max_num_seqs=16,
                    ),
                    root=ROOT,
                    env={"WEIGHT_PATH": "/weights/RWKV"},
                    config=loaded,
                )

            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["exit_code"], 0)
        self.assertEqual(payload["summary"]["status_counts"], {"completed": 1})
        self.assertEqual(payload["runtime"]["wkv_mode"], "fp16")
        self.assertEqual(payload["runtime"]["max_num_seqs"], 16)
        self.assertEqual(payload["slots"], [{"gpu": None, "index": 0, "port": None}])
        self.assertEqual(payload["units"][0]["status"], "completed")
        self.assertEqual(payload["units"][0]["slot_index"], 0)
        self.assertEqual(payload["units"][0]["tasks"], ["gsm8k|0"])


def _postgres_connection() -> dict[str, object] | None:
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        return None
    import asyncio
    import getpass

    async def probe() -> dict[str, object] | None:
        for port in (int(os.environ.get("PGPORT") or 0) or None, 5432, 5433):
            if port is None:
                continue
            kwargs = {
                "user": os.environ.get("PGUSER") or getpass.getuser(),
                "host": os.environ.get("PGHOST") or "/var/run/postgresql",
                "database": os.environ.get("PGDATABASE") or "postgres",
                "port": port,
            }
            try:
                conn = await asyncpg.connect(**kwargs)
            except Exception:
                continue
            await conn.close()
            return kwargs
        return None

    return asyncio.run(probe())


_POSTGRES_KWARGS = _postgres_connection()


@unittest.skipUnless(_POSTGRES_KWARGS, "PostgreSQL is not reachable")
class ScoreboardIngestionTests(unittest.TestCase):
    def test_ingest_records_scores_for_each_task(self) -> None:
        import asyncio
        import uuid

        import asyncpg

        kwargs = dict(_POSTGRES_KWARGS)
        db_name = f"helicopter_scoreboard_test_{uuid.uuid4().hex[:12]}"

        async def create_db() -> None:
            conn = await asyncpg.connect(**kwargs)
            try:
                await conn.execute(f'CREATE DATABASE "{db_name}"')
            finally:
                await conn.close()

        async def drop_db() -> None:
            conn = await asyncpg.connect(**kwargs)
            try:
                await conn.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = $1 AND pid <> pg_backend_pid()
                    """,
                    db_name,
                )
                await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
            finally:
                await conn.close()

        asyncio.run(create_db())
        scoreboard_path = ROOT / "src/scoreboard-server"
        if str(scoreboard_path) not in sys.path:
            sys.path.insert(0, str(scoreboard_path))
        db_env = {
            "SCOREBOARD_DB_HOST": str(kwargs["host"]),
            "SCOREBOARD_DB_PORT": str(kwargs["port"]),
            "SCOREBOARD_DB_USER": str(kwargs["user"]),
            "SCOREBOARD_DB_NAME": db_name,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                results_path = Path(tmp) / "results_2026-07-06T00-00-00.000000.json"
                details_path = Path(tmp) / "details_2026-07-06T00-00-00.000000.parquet"
                results_path.write_text(
                    json.dumps(
                        {
                            "config_general": {"model_name": "openai/unit-test-model"},
                            "results": {
                                "gsm8k|0": {"extractive_match": 0.5},
                                "mmlu|0": {"acc": 0.25},
                                "all": {"extractive_match": 0.5, "acc": 0.25},
                            },
                        }
                    )
                )
                import pandas as pd

                pd.DataFrame(
                    [
                        {
                            "doc": {
                                "id": "gsm8k-0",
                                "task_name": "gsm8k|0",
                                "query": "What is 40 + 2?",
                                "choices": ["42"],
                                "gold_index": [0],
                            },
                            "metric": {"extractive_match": 1.0},
                            "model_response": {
                                "input": "What is 40 + 2?",
                                "text": ["42"],
                                "text_post_processed": ["42"],
                            },
                        },
                        {
                            "doc": {
                                "id": "mmlu-0",
                                "task_name": "mmlu|0",
                                "query": "Choose A.",
                                "choices": ["A", "B"],
                                "gold_index": [0],
                            },
                            "metric": {"acc": 1.0},
                            "model_response": {
                                "input": "Choose A.",
                                "text": ["A"],
                                "text_post_processed": ["A"],
                            },
                        },
                    ]
                ).to_parquet(details_path)
                with mock.patch.dict(os.environ, db_env):
                    from scoreboard_server.db.connection import close_db, init_db
                    from scoreboard_server.db.settings import DatabaseSettings

                    async def prepare_schema() -> None:
                        await init_db(DatabaseSettings.from_env(), generate_schemas=True)
                        await close_db()

                    asyncio.run(prepare_schema())
                    recorded = asyncio.run(
                        eval_run._ingest_scoreboard_results(
                            result_files=[str(results_path)],
                            detail_files=[str(details_path)],
                            model_name="unit-test-model",
                            root=ROOT,
                        )
                    )

                    self.assertEqual(len(recorded), 2)

                    async def read_back() -> list[dict[str, object]]:
                        await init_db(DatabaseSettings.from_env())
                        try:
                            from scoreboard_server.db.repository import ScoreboardStore

                            store = ScoreboardStore(settings=DatabaseSettings.from_env())
                            return await store.list_latest_scores_for_space()
                        finally:
                            await close_db()

                    rows = asyncio.run(read_back())
        finally:
            asyncio.run(drop_db())

        datasets = {str(row.get("dataset") or row.get("benchmark_name")): row for row in rows}
        self.assertIn("gsm8k", datasets)
        self.assertIn("mmlu", datasets)
        gsm8k_metrics = datasets["gsm8k"].get("metrics")
        self.assertEqual(gsm8k_metrics, {"extractive_match": 0.5})


if __name__ == "__main__":
    unittest.main()
