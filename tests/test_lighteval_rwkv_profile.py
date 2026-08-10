from __future__ import annotations

import ast
import json
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

from helicopter_cli import commands, config, eval_run
from lighteval.models.endpoints.litellm_model import LiteLLMClient, LiteLLMModelConfig
from lighteval.models.endpoints.rwkv_profile import (
    RWKV_COMPLETION_SAMPLING_FIELDS,
    RWKVCompletionProfile,
)
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.aime import aime_prompt
from lighteval.tasks.tasks.gpqa import gpqa_instruct_prompt
from lighteval.tasks.tasks.ifeval.main import ifeval_prompt
from lighteval.tasks.tasks.lcb.main import prepare_prompt


ROOT = Path(__file__).resolve().parents[1]
MATH_PROFILE = ROOT / "configs/eval/profiles/g1h-7.2b/math-cot.toml"


def test_all_g1h_category_profiles_are_valid() -> None:
    paths = sorted((MATH_PROFILE.parent).glob("*.toml"))
    assert [path.name for path in paths] == [
        "coding-nocot.toml",
        "instruction-nocot.toml",
        "knowledge-cot.toml",
        "math-cot.toml",
    ]
    profiles = [RWKVCompletionProfile.from_path(path) for path in paths]
    assert {profile.model for profile in profiles} == {"g1h-7.2b"}
    assert {profile.adapter for profile in profiles} == {
        "choice",
        "code",
        "instruction",
        "math",
    }


def test_profile_sampling_schema_matches_vllm_rwkv_completion_request() -> None:
    protocol_path = (
        ROOT
        / "src/infer/vllm-rwkv/vllm/entrypoints/openai/completion/protocol.py"
    )
    module = ast.parse(protocol_path.read_text(encoding="utf-8"))
    completion_request = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "CompletionRequest"
    )
    request_fields = {
        node.target.id
        for node in completion_request.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert RWKV_COMPLETION_SAMPLING_FIELDS <= request_fields
    assert {"stop_tokens", "ban_tokens", "pad_zero", "prefill_chunk_size"}.isdisjoint(
        request_fields
    )


def test_profile_builds_exact_vllm_rwkv_completion_payload() -> None:
    profile = RWKVCompletionProfile.from_path(MATH_PROFILE)
    prompt = profile.render_prompt("What is 1 + 1?")

    assert prompt == "User: What is 1 + 1?\n\nAssistant: <think"
    assert profile.completion_payload(served_model="rwkv7-test", prompt=prompt) == {
        "model": "rwkv7-test",
        "prompt": prompt,
        "n": 8,
        "max_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.35,
        "top_k": 40,
        "presence_penalty": 0.65,
        "frequency_penalty": 0.25,
        "repetition_penalty": 1.0,
        "penalty_decay": 0.99,
        "stop_token_ids": [0],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stop_tokens", "[0]"),
        ("ban_tokens", "[]"),
        ("pad_zero", "true"),
        ("prefill_chunk_size", "16"),
    ),
)
def test_profile_rejects_non_completion_sampling_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        f"""
[profile]
name = "invalid"
model = "g1h-7.2b"
tasks = ["aime24"]
adapter = "math"

[prompt]
mode = "naive_cot"
template = "User: {{query}}\\n\\nAssistant: <think"

[evaluation]
metric = "avg"
num_samples = 8

[sampling]
max_tokens = 4096
temperature = 0.8
{field} = {value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"unsupported \[sampling\] field.*{field}"):
        RWKVCompletionProfile.from_path(path)


def test_source_client_posts_completions_with_profile_rollout_count() -> None:
    profile = RWKVCompletionProfile.from_path(MATH_PROFILE)
    client = object.__new__(LiteLLMClient)
    client.rwkv_profile = profile
    client.model = "openai/rwkv7-test"
    client.base_url = "http://127.0.0.1:8000/v1"
    client.api_key = "EMPTY"
    client.timeout = 30
    client.API_MAX_RETRY = 1
    client.API_RETRY_SLEEP = 0
    client.API_RETRY_MULTIPLIER = 1
    client.concurrent_requests = 1

    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {"index": index, "text": f"> answer {index}", "finish_reason": "stop"}
            for index in range(8)
        ],
        "usage": {"completion_tokens": 16},
    }
    doc = Doc(
        task_name="aime24",
        query="LightEval-added text must not be used",
        raw_query="RAW QUESTION",
        choices=["1"],
        gold_index=0,
    )

    with mock.patch(
        "lighteval.models.endpoints.litellm_model.requests.post",
        return_value=response,
    ) as post:
        outputs = client._rwkv_profile_greedy_until([doc])

    assert post.call_args.args[0] == "http://127.0.0.1:8000/v1/completions"
    payload = post.call_args.kwargs["json"]
    assert payload["prompt"] == "User: RAW QUESTION\n\nAssistant: <think"
    assert payload["n"] == profile.num_samples == doc.num_samples == 8
    assert "messages" not in payload
    assert outputs[0].text == [f"answer {index}" for index in range(8)]


def test_representative_domains_expose_only_raw_dataset_text() -> None:
    aime = aime_prompt({"problem": "ORIGINAL MATH", "answer": "42"}, "aime24")
    assert aime.query == aime.raw_query == "ORIGINAL MATH"

    gpqa_row = {
        "Question": "ORIGINAL QUESTION",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong 1",
        "Incorrect Answer 2": "wrong 2",
        "Incorrect Answer 3": "wrong 3",
    }
    with mock.patch("lighteval.tasks.tasks.gpqa.random.randint", return_value=1):
        gpqa = gpqa_instruct_prompt(gpqa_row, "gpqa:diamond")
    assert gpqa.query == gpqa.raw_query
    assert gpqa.query.startswith("ORIGINAL QUESTION\n\nA) wrong 1")
    assert "Answer the following" not in gpqa.query
    assert "step by step" not in gpqa.query
    assert gpqa.instruction is None

    coding = prepare_prompt(
        {"question_content": "ORIGINAL CODE PROBLEM", "starter_code": "def solve():\n    pass"}
    )
    assert coding == "ORIGINAL CODE PROBLEM\n\ndef solve():\n    pass"
    assert "generate a correct Python" not in coding
    assert "YOUR CODE HERE" not in coding

    instruction = ifeval_prompt(
        {"prompt": "ORIGINAL INSTRUCTION", "instruction_id_list": [], "kwargs": []},
        "ifeval",
    )
    assert instruction.query == instruction.raw_query == "ORIGINAL INSTRUCTION"


def test_cli_uses_profile_tasks_and_disables_runtime_raw_prompt_patch() -> None:
    loaded, _ = config.load_config(ROOT, str(MATH_PROFILE))
    args = Namespace(
        model="g1h-7.2b",
        tasks=None,
        backend="endpoint-litellm",
        config=str(MATH_PROFILE),
        model_args=None,
        extra=None,
    )
    args.tasks = eval_run.resolve_run_tasks(args, loaded)
    assert args.tasks == "aime24,aime25"

    model_args, _ = commands.build_lighteval_model_args(
        args,
        root=ROOT,
        env={},
        config=loaded,
    )
    assert f"rwkv_profile={MATH_PROFILE.resolve()}" in model_args
    assert "generation_parameters" not in model_args
    parsed_model_config = LiteLLMModelConfig.from_args(model_args)
    assert parsed_model_config.rwkv_profile == str(MATH_PROFILE.resolve())

    plan = commands.build_lighteval_plan(
        args,
        root=ROOT,
        env={
            "HELICOPTER_PROMPT_TEMPLATE": "legacy {query}",
            "HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS": "1",
            "HELICOPTER_SCOREBOARD_DB_ONLY": "1",
            "HELICOPTER_VLLM_SAMPLING_JSON": '{"temperature":1.9}',
        },
        config=loaded,
    )
    assert plan.env["HELICOPTER_LIGHTEVAL_PROFILE_PATH"] == str(MATH_PROFILE.resolve())
    assert "HELICOPTER_PROMPT_TEMPLATE" not in plan.env
    assert "HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS" not in plan.env
    assert "HELICOPTER_SCOREBOARD_DB_ONLY" not in plan.env
    assert "HELICOPTER_VLLM_SAMPLING_JSON" not in plan.env
    request_policy = json.loads(plan.env["HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY"])
    assert request_policy["tasks"]["aime24"]["adapter"] == "math"

    args.tasks = "aime24"
    single_task_plan = commands.build_lighteval_plan(
        args, root=ROOT, env={}, config=loaded
    )
    assert single_task_plan.env["HELICOPTER_SCOREBOARD_DB_ONLY"] == "1"
    assert "HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS" not in single_task_plan.env


def test_cli_rejects_profile_sampling_override() -> None:
    loaded, _ = config.load_config(ROOT, str(MATH_PROFILE))
    args = Namespace(
        model="g1h-7.2b",
        model_args=None,
        max_tokens=123,
    )
    with pytest.raises(SystemExit, match="sampling overrides are disabled"):
        commands.build_lighteval_model_args(args, root=ROOT, env={}, config=loaded)
