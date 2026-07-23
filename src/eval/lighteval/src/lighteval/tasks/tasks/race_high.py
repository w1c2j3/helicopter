"""
name:
Race High

dataset:
EleutherAI/race

abstract:
RACE is a large-scale reading comprehension dataset with more than 28,000
passages and nearly 100,000 questions. The dataset is collected from English
examinations in China, which are designed for middle school and high school
students. The dataset can be served as the training and test sets for machine
comprehension.

languages:
english

tags:
multiple-choice, reading-comprehension

paper:
https://aclanthology.org/D17-1082/
"""

import ast

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def race_prompt(line, task_name: str = None):
    problem = ast.literal_eval(line["problems"])[-1]
    query = f"{str(line['article']).strip()}\n\n{str(problem['question']).strip()}"
    return Doc(task_name=task_name, query=query,
               choices=[str(option).strip() for option in problem["options"]],
               gold_index=["A", "B", "C", "D", "E"].index(problem["answer"]), instruction=None)


race_high = LightevalTaskConfig(
    name="race:high",
    prompt_function=race_prompt,
    hf_repo="EleutherAI/race",
    hf_subset="high",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    version=1,
)

TASKS_TABLE = [
    race_high,
]
