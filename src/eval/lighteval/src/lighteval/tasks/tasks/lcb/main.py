"""
name:
Live Code Bench

dataset:
lighteval/code_generation_lite

abstract:
LiveCodeBench collects problems from periodic contests on LeetCode, AtCoder, and
Codeforces platforms and uses them for constructing a holistic benchmark for
evaluating Code LLMs across variety of code-related scenarios continuously over
time.

languages:
english

tags:
code-generation

paper:
https://livecodebench.github.io/

starred:
true
"""

import json
from typing import Any

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import Doc, LightevalTaskConfig
from lighteval.tasks.requests import SamplingMethod
from lighteval.tasks.tasks.lcb.codegen_metrics import (
    codegen_metrics,
    extract_code,
    translate_private_test_cases,
)


def prepare_prompt(line: dict[str, Any]) -> str:
    query = str(line["question_content"]).strip()
    starter_code = str(line.get("starter_code") or "").strip()
    if starter_code:
        query += f"\n\n{starter_code}"
    return query



def lcb_codegeneration_prompt_fn(line, task_name: str = "lcb:codegeneration") -> Doc:
    # For the prompt we need a more general function that can be used tweaked like in:
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
    query = prepare_prompt(line)
    # List of dicts of the form: [{"input": "6\nabc\nacb\nbac\nbca\ncab\ncba\n", "output": "YES\nYES\nYES\nNO\nNO\nYES\n", "testtype": "stdin"}]
    public_test_cases = json.loads(line["public_test_cases"])
    private_test_cases = translate_private_test_cases(line["private_test_cases"])
    inputs = [test["input"] for test in public_test_cases + private_test_cases]
    outputs = [test["output"] for test in public_test_cases + private_test_cases]
    return Doc(
        task_name=task_name,
        query=query,
        choices=[""],
        gold_index=0,
        specific={
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": json.loads(line["metadata"]).get("func_name", None),
        },
    )


class CodegenMetric(SampleLevelComputation):
    def score_one(self, doc: Doc, prediction: str) -> float:
        """Return the pass/fail score for one generated completion.

        ``compute`` keeps the historical multi-sample Pass@1 behavior.  The
        avg@k branch uses this scalar scorer once per completion and lets
        LightEval's official ``AvgAtN`` perform the aggregation.
        """
        assert doc.specific is not None, "Doc specific field is required for codegen_metric"

        evaluation_sample = {
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        metrics, _ = codegen_metrics(
            [{"input_output": json.dumps(evaluation_sample)}],
            [[extract_code(prediction)]],
            k_list=[1],
            num_process_evaluate=8,
        )
        return float(metrics["pass@1"])

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict:
        """Estimates the Pass@1 metric for the code generation task.
        Extract the code from each prediction, Runs it for each sample and generations,
        and computes the Pass@1 over the outputs.
        """
        assert doc.specific is not None, "Doc specific field is required for codegen_metric"

        predictions = model_response.final_text
        # Extract generated code snippets
        generated_code_snippets = [[extract_code(pred) for pred in predictions]]  # noqa: F841
        evaluation_sample = {  # noqa: F841
            "inputs": doc.specific["inputs"],
            "outputs": doc.specific["outputs"],
            "fn_name": doc.specific["fn_name"],
        }
        # This is a list of lists because
        evaluation_sample = [{"input_output": json.dumps(evaluation_sample)}]

        metrics, _ = codegen_metrics(
            evaluation_sample,
            generated_code_snippets,
            k_list=[1],  # Only run for Pass@1
            num_process_evaluate=8,
        )
        return metrics["pass@1"]


lcb_codegen_metric = SampleLevelMetric(
    metric_name="codegen_pass@1:16",  # This is the way of informing the number of generations currently
    category=SamplingMethod.GENERATIVE,
    higher_is_better=True,
    sample_level_fn=CodegenMetric(),
    corpus_level_fn=np.mean,
    batched_compute=False,
)


extend_enum(Metrics, "lcb_codegen_metric", lcb_codegen_metric)

configs = [
    "release_v1",
    "release_v2",
    "release_v3",
    "release_v4",
    "release_v5",
    "release_v6",
    "release_latest",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v1_v2",
    "v1_v3",
    "v1_v4",
    "v1_v5",
    "v2_v3",
    "v2_v4",
    "v2_v5",
    "v3_v4",
    "v3_v5",
    "v4_v5",
]

tasks = []

for subset in configs:
    # To keep the base subset as the default, the others are named "lcb:codegeneration_v4", "lcb:codegeneration_v5"... etc
    name = "lcb:codegeneration" if subset == "v4_v5" else f"lcb:codegeneration_{subset}"
    task = LightevalTaskConfig(
        name=name,
        prompt_function=lcb_codegeneration_prompt_fn,
        hf_repo="lighteval/code_generation_lite",
        hf_subset=subset,  # https://github.com/LiveCodeBench/LiveCodeBench/tree/main?tab=readme-ov-file#dataset-versions
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        generation_size=32768,
        metrics=[Metrics.lcb_codegen_metric],
        stop_sequence=[],  # no stop sequence, will use EOS token
        version=1,
    )
    tasks.append(task)


TASKS_TABLE = tasks
