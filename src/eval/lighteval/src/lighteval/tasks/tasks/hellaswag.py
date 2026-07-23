"""
name:
Hellaswag

dataset:
Rowan/hellaswag

abstract:
HellaSwag is a commonsense inference benchmark designed to challenge language
models with adversarially filtered multiple-choice questions.

languages:
english

tags:
multiple-choice, narrative, reasoning

paper:
https://arxiv.org/abs/1905.07830
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def hellaswag_prompt(line, task_name: str = None):
    query = f"{str(line['activity_label']).strip()}: {str(line['ctx_a']).strip()} {str(line['ctx_b']).strip()}"
    query += "".join(f"\n{key}. {str(choice).strip()}" for key, choice in zip(ascii_uppercase, line["endings"]))
    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(task_name=task_name, query=query,
               choices=[" " + label for label in ascii_uppercase[: len(line["endings"])]],
               gold_index=gold_ix, instruction=None)


hellaswag = LightevalTaskConfig(
    name="hellaswag",
    prompt_function=hellaswag_prompt,
    hf_repo="Rowan/hellaswag",
    hf_subset="default",
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
    hellaswag,
]
