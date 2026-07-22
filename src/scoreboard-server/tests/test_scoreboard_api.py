from __future__ import annotations

import getpass
import os
import uuid

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from scoreboard_server.db.connection import close_db, init_db
from scoreboard_server.db.lease import SchedulerLeaseManager, SchedulerLeaseStore
from scoreboard_server.db.settings import DatabaseSettings
from scoreboard_server.application import create_app
from scoreboard_server.db.repository import ScoreboardStore
from scoreboard_server.db.models import Benchmark, BenchmarkCatalog, Completion, Task


def _maintenance_connection_kwargs() -> dict[str, str]:
    return {
        "user": os.environ.get("PGUSER") or getpass.getuser(),
        "host": os.environ.get("PGHOST") or "/var/run/postgresql",
        "database": os.environ.get("PGDATABASE") or "postgres",
    }


@pytest.fixture()
async def database_settings() -> DatabaseSettings:
    db_name = f"helicopter_scoreboard_test_{uuid.uuid4().hex[:12]}"
    kwargs = _maintenance_connection_kwargs()
    conn = await asyncpg.connect(**kwargs)
    try:
        await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()

    settings = DatabaseSettings(
        host=kwargs["host"],
        port=int(os.environ.get("PGPORT") or 5432),
        user=kwargs["user"],
        password=os.environ.get("PGPASSWORD") or None,
        database=db_name,
    )
    await init_db(settings, generate_schemas=True)
    try:
        yield settings
    finally:
        await close_db()
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


async def _seed_scoreboard(settings: DatabaseSettings) -> int:
    service = ScoreboardStore(settings=settings)
    await service.ensure_benchmark_num_samples(dataset="gsm8k_test", num_samples=2)
    task_id = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={
            "avg_k": 1,
            "pass_ks": [1],
            "prompt_profile": "naive",
            "sampling_config": {
                "answer": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 128,
                    "stop_tokens": [0],
                }
            },
        },
        allow_resume=True,
    )
    await service.insert_completion_payloads_batch(
        task_id=task_id,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+1?",
                "completion1": "2",
                "stop_reason1": "stop",
                "sampling_config": {"answer": {"temperature": 0.2, "stop_tokens": [0]}},
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+2?",
                "completion1": "4",
                "stop_reason1": "stop",
                "sampling_config": {"answer": {"temperature": 0.2, "stop_tokens": [0]}},
            },
        ],
    )
    inserted = await service.ingest_eval_payloads(
        task_id=task_id,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "2",
                "ref_answer": "2",
                "is_passed": True,
                "fail_reason": "",
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "4",
                "ref_answer": "3",
                "is_passed": False,
                "fail_reason": "wrong arithmetic",
            },
        ],
    )
    assert inserted == 2
    await service.record_score_payload(
        task_id=task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"avg@1": 0.5}, "created_at": "2026-07-01T12:00:00"},
    )
    return int(task_id)


@pytest.mark.asyncio
async def test_exact_catalog_name_is_not_split_as_suffix(database_settings: DatabaseSettings) -> None:
    await BenchmarkCatalog.create(
        benchmark_name="human_eval",
        benchmark_split="",
        field="coding",
        source="lighteval",
        source_family="lighteval",
        target_kind="task",
        run_status="direct_lighteval_task",
        scope="test",
        metadata=None,
    )
    store = ScoreboardStore(settings=database_settings)
    task_id = await store.get_or_create_task(
        job_name="judge",
        job_id=None,
        dataset="human_eval",
        model="rwkv7-test",
        is_param_search=False,
        allow_resume=False,
    )
    task = await Task.get(task_id=int(task_id))
    benchmark = await Benchmark.get(benchmark_id=task.benchmark_id)
    assert benchmark.benchmark_name == "human_eval"
    assert benchmark.benchmark_split == ""


async def test_scoreboard_api_serves_leaderboard_records_context_and_history(
    database_settings: DatabaseSettings,
) -> None:
    task_id = await _seed_scoreboard(database_settings)
    service = ScoreboardStore(settings=database_settings)
    tuning_task_id = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-tuning",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=True,
        sampling_config={
            "prompt_profile": "normal",
            "sampling_config": {"answer": {"temperature": 0.7, "top_k": 40, "top_p": 0.9}},
        },
        allow_resume=False,
    )
    await service.record_score_payload(
        task_id=tuning_task_id,
        payload={"cot_mode": "CoT", "metrics": {"accuracy": 0.6}, "created_at": "2026-07-01T13:00:00"},
    )
    app = create_app(settings=database_settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        meta = (await client.get("/api/meta")).json()
        assert meta["entry_count"] == 1
        assert "rwkv7-g1g-1.5b" in meta["models"]
        assert any(group["key"] == "math" for group in meta["domain_groups"])

        leaderboard = (
            await client.get("/api/leaderboard", params={"model": "rwkv7-g1g-1.5b", "view": "benchmark_detail_latest"})
        ).json()
        math_domain = next(domain for domain in leaderboard["domains"] if domain["key"] == "math")
        assert math_domain["rows"][0]["benchmark_name"] == "gsm8k_test"
        assert math_domain["rows"][0]["cells"][0]["percent"] == 50.0
        assert math_domain["rows"][0]["cells"][0]["meta"]["task_id"] == task_id

        all_matrix = next(domain for domain in leaderboard["matrix"]["domains"] if domain["key"] == "all")
        assert all_matrix["columns"][0]["key"] == "gsm8k_test"
        assert all_matrix["rows"][0]["model"] == "rwkv7-g1g-1.5b"
        assert all_matrix["rows"][0]["cells"][0]["percent"] == 50.0

        tuning = leaderboard["tuning_matrix"]["benchmarks"][0]
        assert tuning["key"] == "gsm8k_test"
        assert tuning["columns"][0]["label"] == "normal · CoT · T0.7 · K40 · P0.9"
        assert tuning["rows"][0]["best"] == 60.0
        assert tuning["rows"][0]["cells"][0]["meta"]["task_id"] == int(tuning_task_id)

        records = (await client.get("/api/eval-records", params={"task_id": task_id, "limit": 10})).json()
        assert len(records["records"]) == 2
        assert records["records"][1]["fail_reason"] == "wrong arithmetic"

        wrong = (
            await client.get("/api/eval-records", params={"task_id": task_id, "only_wrong": "true", "limit": 10})
        ).json()
        assert [row["sample_index"] for row in wrong["records"]] == [1]

        context = (
            await client.get(
                "/api/eval-context",
                params={"task_id": task_id, "sample_index": 0, "repeat_index": 0, "pass_index": 0},
            )
        ).json()
        assert context["view"] == "structured"
        assert context["context"]["stages"][0]["prompt"] == "What is 1+1?"
        assert context["stop_tokens"]["answer"][0]["id"] == 0

        options = (await client.get("/api/score-history/options")).json()
        assert {"model": "rwkv7-g1g-1.5b", "dataset": "gsm8k_test"} in options["pairs"]

        history = (
            await client.get("/api/score-history", params={"model": "rwkv7-g1g-1.5b", "benchmark": "gsm8k_test"})
        ).json()
        assert history["total"] == 1
        assert history["groups"][0]["points"][0]["percent"] == 50.0

        detail = (await client.get("/api/score-history/detail", params={"task_id": task_id})).json()
        assert detail["found"] is True
        assert detail["metric"] == "avg@1"
        assert detail["sampling"]["stages"]["answer"]["temperature"] == 0.2


async def test_leaderboard_exposes_upstream_coding_chart_payload(
    database_settings: DatabaseSettings,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    await service.ensure_benchmark_num_samples(dataset="humaneval_test", num_samples=164)
    task_id = await service.get_or_create_task(
        job_name="code_generation",
        job_id="job-code",
        dataset="humaneval_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"pass_ks": [1]},
        allow_resume=False,
    )
    await service.record_score_payload(
        task_id=task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"pass@1": 0.25}, "created_at": "2026-07-01T12:00:00"},
    )
    app = create_app(settings=database_settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={"model": "rwkv7-g1g-1.5b", "view": "benchmark_detail_latest"},
            )
        ).json()

    chart = leaderboard["charts"]["coding"]
    assert chart["type"] == "coding_bar"
    assert chart["datasets"] == ["HUMANEVAL"]
    assert chart["models"] == ["rwkv7-g1g-1.5b"]
    assert chart["data"] == [
        {
            "dataset": "HUMANEVAL",
            "model": "rwkv7-g1g-1.5b",
            "score": 0.25,
            "metric": "pass@1",
        }
    ]


async def test_leaderboard_exposes_upstream_non_coding_chart_payloads(
    database_settings: DatabaseSettings,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    chart_cases = [
        (
            "mmlu_test",
            "free_response",
            {
                "cot_mode": "NoCoT",
                "metrics": {"accuracy_by_subject": {"physics": 0.8, "history": 0.6}},
                "created_at": "2026-07-01T12:00:00",
            },
        ),
        (
            "aime24_test",
            "free_response",
            {
                "cot_mode": "NoCoT",
                "metrics": {"pass@1": 0.2, "pass@4": 0.35},
                "created_at": "2026-07-01T12:01:00",
            },
        ),
        (
            "ifeval_test",
            "instruction_following",
            {
                "cot_mode": "NoCoT",
                "metrics": {
                    "tier0_accuracy": {"keywords:include": 0.75, "language:zh": 0.5},
                    "tier1_accuracy": {"prompt:overall": 1.0},
                },
                "created_at": "2026-07-01T12:02:00",
            },
        ),
    ]
    for dataset, job_name, score_payload in chart_cases:
        await service.ensure_benchmark_num_samples(dataset=dataset, num_samples=1)
        task_id = await service.get_or_create_task(
            job_name=job_name,
            job_id=f"job-{dataset}",
            dataset=dataset,
            model="rwkv7-g1g-1.5b",
            is_param_search=False,
            sampling_config={},
            allow_resume=False,
        )
        await service.record_score_payload(task_id=task_id, payload=score_payload)

    app = create_app(settings=database_settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={"model": "rwkv7-g1g-1.5b", "view": "benchmark_detail_latest"},
            )
        ).json()

    knowledge = leaderboard["charts"]["knowledge"]
    assert knowledge["type"] == "knowledge_bar"
    assert knowledge["subjects"] == ["Physics", "History"]
    assert {row["subject"]: row["score"] for row in knowledge["data"]} == {"Physics": 0.8, "History": 0.6}

    math = leaderboard["charts"]["math"]
    assert math["type"] == "aime_line"
    assert math["ks"] == [1, 4]
    assert math["series"] == [
        {
            "name": "AIME24 · rwkv7-g1g-1.5b",
            "points": [{"k": 1, "acc": 0.2}, {"k": 4, "acc": 0.35}],
        }
    ]

    instruction = leaderboard["charts"]["instruction_following"]
    assert instruction["type"] == "instruction_bar"
    assert instruction["domains"] == ["prompt", "keywords", "language"]
    assert {(row["domain"], row["score"]) for row in instruction["data"]} == {
        ("prompt", 1.0),
        ("keywords", 0.75),
        ("language", 0.5),
    }


async def test_delta_detail_rows_keep_cot_and_nocot_benchmarks_adjacent(
    database_settings: DatabaseSettings,
) -> None:
    service = ScoreboardStore(settings=database_settings)

    async def record_pair(
        *,
        dataset: str,
        cot_mode: str,
        prev_accuracy: float,
        latest_accuracy: float,
    ) -> None:
        task_name = "multi_choice_cot" if cot_mode == "CoT" else "multi_choice_plain"
        for model, accuracy, created_at in (
            ("rwkv7-g1f-1.5b-20260526-ctx8192", prev_accuracy, "2026-05-26T12:00:00"),
            ("rwkv7-g1g-1.5b-20260527-ctx8192", latest_accuracy, "2026-05-27T12:00:00"),
        ):
            await service.ensure_benchmark_num_samples(dataset=dataset, num_samples=100)
            task_id = await service.get_or_create_task(
                job_name=task_name,
                job_id=f"{task_name}-{dataset}-{model}",
                dataset=dataset,
                model=model,
                is_param_search=False,
                sampling_config={"cot_mode": cot_mode},
                allow_resume=False,
            )
            await service.record_score_payload(
                task_id=task_id,
                payload={"cot_mode": cot_mode, "metrics": {"accuracy": accuracy}, "created_at": created_at},
            )

    await record_pair(dataset="mmlu_test", cot_mode="CoT", prev_accuracy=0.40, latest_accuracy=0.41)
    await record_pair(dataset="mmlu_test", cot_mode="NoCoT", prev_accuracy=0.50, latest_accuracy=0.51)
    await record_pair(dataset="supergpqa_test", cot_mode="CoT", prev_accuracy=0.20, latest_accuracy=0.90)
    app = create_app(settings=database_settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        leaderboard = (
            await client.get(
                "/api/leaderboard",
                params={"model": "每档最新（调度策略）", "view": "benchmark_detail_delta"},
            )
        ).json()

    knowledge = next(domain for domain in leaderboard["domains"] if domain["key"] == "knowledge")
    row_keys = [(row["benchmark_name"], row["eval_method"]) for row in knowledge["rows"]]
    assert row_keys[:3] == [
        ("mmlu_test", "cot"),
        ("mmlu_test", "nocot"),
        ("supergpqa_test", "cot"),
    ]
    assert knowledge["rows"][2]["cells"][0]["delta"] == 70.0


async def test_resume_reuses_running_task_and_upserts_completion(
    database_settings: DatabaseSettings,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    first = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 1},
        allow_resume=True,
    )
    second = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 1},
        allow_resume=True,
    )
    assert second == first

    old_payload = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "prompt1": "old",
        "completion1": "old",
        "sampling_config": {"generation_size": 2048, "effective_generation_size": 4096},
    }
    await service.insert_completion_payloads_batch(task_id=first, payloads=[old_payload])
    original = await Completion.get(
        task_id=int(first),
        sample_index=0,
        avg_repeat_index=0,
        pass_index=0,
    )

    new_payload = {
        **old_payload,
        "prompt1": "new",
        "completion1": "new",
        "sampling_config": {"generation_size": 2048},
    }
    await service.insert_completion_payloads_batch(task_id=first, payloads=[new_payload])
    updated = await Completion.get(completions_id=original.completions_id)

    assert await service.count_completions(task_id=first, status="Completed") == 1
    assert updated.created_at == original.created_at
    payloads = await service.list_completion_payloads(task_id=first, status="Completed")
    assert payloads[0]["prompt1"] == "new"
    assert payloads[0]["sampling_config"]["effective_generation_size"] == 4096

async def test_resume_ignores_git_hash_for_uploaded_running_task(
    database_settings: DatabaseSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    monkeypatch.setenv("RWKV_GIT_SHA", "unknown")
    first = await service.get_or_create_task(
        job_name="lighteval",
        job_id=None,
        dataset="mmlu_anatomy",
        model="rwkv7-test",
        is_param_search=False,
        sampling_config={"avg_k": 8},
        allow_resume=True,
    )

    monkeypatch.setenv("RWKV_GIT_SHA", "local-upload-20260721")
    resumed = await service.get_or_create_task(
        job_name="lighteval",
        job_id=None,
        dataset="mmlu_anatomy",
        model="rwkv7-test",
        is_param_search=False,
        sampling_config={"avg_k": 8},
        allow_resume=True,
    )

    assert resumed == first


async def test_partial_score_remains_resumable_and_eval_is_upserted(
    database_settings: DatabaseSettings,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    task_id = await service.get_or_create_task(
        job_name="lighteval",
        job_id=None,
        dataset="mmlu_anatomy",
        model="rwkv7-test",
        is_param_search=False,
        sampling_config={"avg_k": 8},
        allow_resume=False,
    )
    await service.insert_completion_payloads_batch(
        task_id=task_id,
        payloads=[
            {
                "_stage": "generation",
                "status": "Completed",
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "question",
                "completion1": "answer",
                "sampling_config": {},
            }
        ],
    )
    first_eval = {
        "sample_index": 0,
        "repeat_index": 0,
        "pass_index": 0,
        "answer": "wrong",
        "ref_answer": "right",
        "is_passed": False,
    }
    assert await service.ingest_eval_payloads(task_id=task_id, payloads=[first_eval]) == 1
    second_eval = {**first_eval, "answer": "right", "is_passed": True}
    assert await service.ingest_eval_payloads(task_id=task_id, payloads=[second_eval]) == 1
    eval_rows = await service.list_eval_rows(task_id=task_id)
    assert len(eval_rows) == 1
    assert eval_rows[0]["is_passed"] is True

    await service.record_score_payload(
        task_id=task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"avg@8": 1.0}},
        mark_completed=False,
    )
    bundle = await service.get_task_bundle(task_id=task_id)
    assert bundle is not None
    assert bundle["task"]["status"] == "Running"

    resumed = await service.get_or_create_task(
        job_name="lighteval",
        job_id=None,
        dataset="mmlu_anatomy",
        model="rwkv7-test",
        is_param_search=False,
        sampling_config={"avg_k": 8},
        allow_resume=True,
    )
    assert resumed == task_id

    await service.record_score_payload(
        task_id=task_id,
        payload={"cot_mode": "NoCoT", "metrics": {"avg@8": 1.0, "judge_avg@8": 1.0}},
    )
    new_task_id = await service.get_or_create_task(
        job_name="lighteval",
        job_id=None,
        dataset="mmlu_anatomy",
        model="rwkv7-test",
        is_param_search=False,
        sampling_config={"avg_k": 8},
        allow_resume=True,
    )
    assert new_task_id != task_id



async def test_ref_answer_fallback_and_task_identity_metadata(
    database_settings: DatabaseSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ScoreboardStore(settings=database_settings)
    first = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 1},
        config_path="configs/a.yaml",
        allow_resume=True,
    )
    same_config = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 1},
        config_path="configs/a.yaml",
        allow_resume=True,
    )
    different_config = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 1},
        config_path="configs/b.yaml",
        allow_resume=True,
    )
    assert same_config == first
    assert different_config != first
    bundle = await service.get_task_bundle(task_id=first)
    assert bundle is not None
    assert bundle["task"]["config_path"] == "configs/a.yaml"

    monkeypatch.setenv("RWKV_TASK_DESC", "temporary task")
    monkeypatch.setenv("RWKV_TASK_IS_TMP", "1")
    monkeypatch.setenv("RWKV_SKILLS_LOG_PATH", "/tmp/helicopter.log")
    tmp_task = await service.get_or_create_task(
        job_name="free_response",
        job_id="job-1",
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
        sampling_config={"avg_k": 2},
        config_path="configs/tmp.yaml",
        allow_resume=False,
    )
    tmp_bundle = await service.get_task_bundle(task_id=tmp_task)
    assert tmp_bundle is not None
    assert tmp_bundle["task"]["is_tmp"] is True
    assert tmp_bundle["task"]["desc"] == "temporary task"
    assert tmp_bundle["task"]["log_path"] == "/tmp/helicopter.log"

    await service.insert_completion_payloads_batch(
        task_id=first,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+1?",
                "completion1": "2",
                "sampling_config": {},
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "prompt1": "What is 1+2?",
                "completion1": "3",
                "sampling_config": {},
            },
        ],
    )
    inserted = await service.ingest_eval_payloads(
        task_id=first,
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "2",
                "expected_answer": "2",
                "is_passed": True,
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "answer": "3",
                "raw_record": {"answer": 3},
                "is_passed": True,
            },
        ],
    )
    assert inserted == 2
    eval_rows = await service.list_eval_rows(task_id=first)
    assert [row["ref_answer"] for row in eval_rows] == ["2", "3"]


async def test_service_exposes_upstream_database_operation_surface(
    database_settings: DatabaseSettings,
) -> None:
    task_id = await _seed_scoreboard(database_settings)
    service = ScoreboardStore(settings=database_settings)

    checker_inserted = await service.ingest_checker_payloads(
        task_id=str(task_id),
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "answer_correct": True,
                "reason": "ok",
            },
            {
                "sample_index": 1,
                "repeat_index": 0,
                "pass_index": 0,
                "math_error": True,
                "needs_human_review": True,
                "reason": "wrong arithmetic",
            },
        ],
    )
    assert checker_inserted == 2
    assert await service.list_checker_keys(task_id=str(task_id)) == {(0, 0, 0), (1, 0, 0)}

    completion_rows = await service.list_completions_rows(task_id=str(task_id))
    eval_rows = await service.list_eval_rows(task_id=str(task_id))
    checker_rows = await service.list_checker_rows(task_id=str(task_id))
    score_rows = await service.list_scores_rows(task_id=str(task_id))
    assert [row["sample_index"] for row in completion_rows] == [0, 1]
    assert {row["completions_id"] for row in eval_rows} == {row["completions_id"] for row in completion_rows}
    assert checker_rows[1]["needs_human_review"] is True
    assert score_rows[0]["metrics"] == {"avg@1": 0.5}

    score_payload = await service.get_score_payload(task_id=str(task_id))
    assert score_payload is not None
    assert score_payload["task_id"] == task_id
    assert score_payload["model"] == "rwkv7-g1g-1.5b"

    official_scores = await service.list_scores_by_dataset(
        dataset="gsm8k_test",
        model="rwkv7-g1g-1.5b",
        is_param_search=False,
    )
    assert [row["task_id"] for row in official_scores] == [task_id]

    progress = await service.get_latest_task_generation_progress(
        evaluator="free_response",
        model_name="rwkv7-g1g-1.5b",
        benchmark_name="gsm8k",
        benchmark_split="test",
    )
    assert progress == {
        "task_id": task_id,
        "status": "Completed",
        "sampling_config": {
            "avg_k": 1,
            "pass_ks": [1],
            "prompt_profile": "normal",
            "sampling_config": {
                "answer": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 128,
                    "stop_tokens": [0],
                }
            },
        },
        "completed_completions": 2,
        "total_completions": 2,
        "has_score": True,
    }

    strategy_task_ids = await service.ingest_eval_payload_groups(
        task_id=str(task_id),
        completion_payloads=await service.list_completion_payloads(task_id=str(task_id), status="Completed"),
        payloads_by_group={
            "primary": [
                {
                    "sample_index": 0,
                    "repeat_index": 0,
                    "pass_index": 0,
                    "answer": "2",
                    "ref_answer": "2",
                    "is_passed": True,
                    "fail_reason": "",
                }
            ],
            "judge_v2": [
                {
                    "sample_index": 0,
                    "repeat_index": 0,
                    "pass_index": 0,
                    "answer": "2",
                    "ref_answer": "2",
                    "is_passed": True,
                    "fail_reason": "",
                }
            ],
        },
        primary_group="primary",
    )
    assert strategy_task_ids["primary"] == task_id
    strategy_task_id = strategy_task_ids["judge_v2"]
    strategy_bundle = await service.get_task_bundle(task_id=str(strategy_task_id))
    assert strategy_bundle is not None
    assert strategy_bundle["task"]["is_tmp"] is True
    assert strategy_bundle["task"]["is_param_search"] is True
    assert strategy_bundle["task"]["status"] == "Completed"
    assert "eval_strategy=judge_v2" in strategy_bundle["task"]["desc"]
    assert len(await service.list_eval_rows(task_id=str(strategy_task_id))) == 1


async def test_scheduler_lease_store_keeps_foreign_active_jobs(
    database_settings: DatabaseSettings,
) -> None:
    store = SchedulerLeaseStore(settings=database_settings)
    first = SchedulerLeaseManager(store, node_id="node-a", owner_id="owner-a", lease_duration_s=30)
    second = SchedulerLeaseManager(store, node_id="node-b", owner_id="owner-b", lease_duration_s=30)

    assert await first.claim("job-1", lease_meta={"task_id": 1}) is True
    assert await second.claim("job-1") is False
    assert await first.renew(["job-1", "missing"]) == {"job-1"}
    assert await second.active_foreign_job_ids() == {"job-1"}
    assert await first.release(["job-1"]) == 1
    assert await second.active_foreign_job_ids() == set()


async def test_admin_routes_expose_live_scheduler_contract(
    database_settings: DatabaseSettings,
) -> None:
    app = create_app(settings=database_settings)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        health = (await client.get("/api/admin/health")).json()
        assert health == {"status": "ok", "active": False, "auth_required": False}

        options = (await client.get("/api/admin/eval/options")).json()
        assert "g1g-1.5b" in options["model_select"]
        assert "configs/example.toml" in options["configs"]

        draft = (await client.get("/api/admin/eval/draft")).json()
        assert draft["config"] == "configs/example.toml"
        assert draft["models"] == ["g1g-1.5b"]
        assert draft["scoreboard"] is True

        status = (await client.get("/api/admin/eval/status")).json()
        assert status["status"] == "idle"
        assert status["error"] is None
        assert status["queue_head"] == []
        assert status["available_gpus"] == []

        assert (await client.post("/api/admin/eval/pause")).status_code == 409
        assert (await client.post("/api/admin/eval/resume")).status_code == 409
        assert (await client.post("/api/admin/eval/cancel")).status_code == 409

        backpressure = (await client.get("/api/admin/backpressure")).json()
        assert backpressure == {
            "infer_base_url": "",
            "available_gpus": [],
            "models": [],
            "error": None,
        }
