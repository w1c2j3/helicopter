"""
name:
Gsm8K

dataset:
openai/gsm8k

abstract:
GSM8K is a dataset of 8,000+ high-quality, single-step arithmetic word problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2110.14168
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


MATH_PROMPT_TEMPLATE = "{prompt}"


def gsm8k_final_answer(answer: str) -> str:
    parts = str(answer).rsplit("####", 1)
    if len(parts) != 2 or not parts[1].strip():
        raise ValueError("GSM8K row has no final answer after ####")
    return parts[1].strip()


def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = gsm8k_final_answer(record["answer"])
    answer.pop()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample):
    return f"{sample.input}\n\nReasoning:\n" + f"{sample.metadata['reasoning']}\n\n" + f"ANSWER: {sample.target}"


def gsm8k_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=str(line["question"]).strip().replace("\r\n", "\n"),
        choices=[f"$\\boxed{{{gsm8k_final_answer(line['answer'])}}}$"],
        gold_index=0,
    )


gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:"],
    version=1,
)

TASKS_TABLE = [
    gsm8k,
]
