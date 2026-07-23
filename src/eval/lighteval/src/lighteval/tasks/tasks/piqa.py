"""
name:
Piqa

dataset:
ybisk/piqa

abstract:
PIQA is a benchmark for testing physical commonsense reasoning. It contains
questions requiring this kind of physical commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
https://arxiv.org/abs/1911.11641
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def piqa_prompt(line, task_name: str = None):
    letters = list(ascii_uppercase)[:2]
    solutions = [str(line["sol1"]).strip(), str(line["sol2"]).strip()]
    query = str(line["goal"]).strip()
    query += "".join(f"\n{key}. {choice}" for key, choice in zip(letters, solutions))
    return Doc(task_name=task_name, query=query, choices=letters,
               gold_index=int(line["label"]), instruction=None)


piqa = LightevalTaskConfig(
    name="piqa",
    prompt_function=piqa_prompt,
    # Script-free official LightEval mirror. Its three parquet splits are
    # row-for-row identical to the original ybisk/piqa parquet revision.
    hf_repo="lighteval/piqa",
    hf_subset="plain_text",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    piqa,
]
