from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from helicopter_lm_eval import existing
from helicopter_lm_eval import __main__ as entrypoint
from helicopter_lm_eval.existing import ExistingPublicationError


CAMPAIGN_ID = "11111111-1111-1111-1111-111111111111"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_unit(
    output_dir: Path,
    *,
    mode: str = "fp16",
    task_names: tuple[str, ...] = ("wikitext",),
    weight_sha256: str | None = "a" * 64,
) -> Path:
    output_dir.mkdir(parents=True)
    results_by_task: dict[str, object] = {}
    samples_by_task: dict[str, object] = {}
    configs_by_task: dict[str, object] = {}
    counts_by_task: dict[str, object] = {}
    versions: dict[str, object] = {}
    benchmark_artifacts: list[dict[str, object]] = []
    for task_name in task_names:
        results_by_task[task_name] = {
            "alias": task_name,
            "acc,none": 1.0,
            "acc_stderr,none": 0.0,
        }
        samples_by_task[task_name] = [
            {
                "doc_id": 0,
                "doc": {"question": "2 + 2?", "answer": "4"},
                "filter": "none",
                "metrics": ["acc"],
                "acc": 1.0,
                "filtered_resps": ["4"],
                "resps": [["4"]],
            }
        ]
        configs_by_task[task_name] = {
            "task": task_name,
            "dataset_path": f"fixture/{task_name}",
            "dataset_name": "default",
            "test_split": "test",
            "output_type": "multiple_choice",
            "metadata": {
                "version": 1.0,
                "config_source": f"/fixture/tasks/{task_name}/{task_name}.yaml",
                "languages": ["en"],
                "tags": ["fixture"],
            },
        }
        counts_by_task[task_name] = {"original": 1, "effective": 1}
        versions[task_name] = 1.0
        task_dir = output_dir / "benchmarks" / task_name
        task_dir.mkdir(parents=True)
        records_path = task_dir / "records.jsonl"
        records_path.write_text("{}\n", encoding="utf-8")
        benchmark_artifacts.append(
            {
                "task_name": task_name,
                "records_path": records_path.relative_to(output_dir).as_posix(),
            }
        )

    _write_json(
        output_dir / "results.json",
        {
            "results": results_by_task,
            "samples": samples_by_task,
            "configs": configs_by_task,
            "n-samples": counts_by_task,
            "versions": versions,
            "config": {"batch_size": 8},
            "date": "2026-08-01T12:00:00Z",
        },
    )
    _write_json(
        output_dir / "summary.json",
        {
            "tasks": list(task_names),
            "wkv_mode": mode,
            "weight_sha256": weight_sha256,
            "weight_display_name": "fixture.pth",
            "model_id": "fixture-model",
            "max_model_len": 8192,
            "eot_token_id": None,
            "date": "2026-08-01T12:00:00Z",
            "prompt": {"profile": "none"},
        },
    )
    _write_json(
        output_dir / "artifacts.json",
        {
            "schema_version": 2,
            "evaluator": {"name": "lm-eval", "version": "0.4.12"},
            "results_path": "results.json",
            "summary_path": "summary.json",
            "benchmark_artifacts": benchmark_artifacts,
        },
    )
    return output_dir


def test_load_existing_unit_preserves_native_metrics_and_samples(
    tmp_path: Path,
) -> None:
    unit = existing.load_existing_unit(_write_unit(tmp_path / "unit"))
    task = existing._unit_tasks(unit)[0]

    payload = existing._task_payload(
        unit=unit,
        task=task,
        campaign_id=CAMPAIGN_ID,
    )

    assert task["identity"] == f"{'a' * 64}:fp16:wikitext"
    assert payload["primary_metric"] == "acc,none"
    assert payload["aggregates"]["acc,none"] == 1.0
    assert payload["details"][0]["model_response"]["filtered_resps"] == ["4"]
    assert payload["artifact"]["details_paths"] == [
        "benchmarks/wikitext/records.jsonl"
    ]
    assert payload["sampling_config"]["source_artifacts"] == {
        "results_sha256": unit.results_sha256,
        "summary_sha256": unit.summary_sha256,
        "artifacts_sha256": unit.artifact_sha256,
        "timestamp": "2026-08-01T12:00:00Z",
    }


def test_existing_campaign_identity_does_not_depend_on_local_path(
    tmp_path: Path,
) -> None:
    first_dir = _write_unit(tmp_path / "first")
    second_dir = tmp_path / "moved"
    shutil.copytree(first_dir, second_dir)
    first = existing.load_existing_unit(first_dir)
    second = existing.load_existing_unit(second_dir)

    first_payload = existing._campaign_payload(
        units=[first],
        expected_tasks=existing._unit_tasks(first),
    )
    second_payload = existing._campaign_payload(
        units=[second],
        expected_tasks=existing._unit_tasks(second),
    )

    assert first_payload["run_key"] == second_payload["run_key"]
    assert first_payload["config_digest"] == second_payload["config_digest"]


def test_existing_artifact_requires_source_records_and_weight_identity(
    tmp_path: Path,
) -> None:
    missing_records = _write_unit(tmp_path / "missing-records")
    (missing_records / "benchmarks/wikitext/records.jsonl").unlink()
    with pytest.raises(ExistingPublicationError, match="does not exist"):
        existing.load_existing_unit(missing_records)

    missing_weight = _write_unit(tmp_path / "missing-weight", weight_sha256=None)
    with pytest.raises(ExistingPublicationError, match="pass --weight-sha256"):
        existing.load_existing_unit(missing_weight)

    loaded = existing.load_existing_unit(
        missing_weight,
        weight_sha256="b" * 64,
        weight_display_name="historical-fixture.pth",
    )
    assert loaded.weight_sha256 == "b" * 64
    assert loaded.weight_display_name == "historical-fixture.pth"


def test_existing_matrix_requires_the_same_task_set(tmp_path: Path) -> None:
    fp16 = existing.load_existing_unit(
        _write_unit(tmp_path / "fp16", task_names=("wikitext",))
    )
    fp32 = existing.load_existing_unit(
        _write_unit(
            tmp_path / "fp32",
            mode="fp32io16",
            task_names=("wikitext", "lambada_openai"),
        )
    )
    units = [fp16, fp32]
    tasks = {unit.output_dir: existing._unit_tasks(unit) for unit in units}

    with pytest.raises(ExistingPublicationError, match="same task set"):
        existing._validate_matrix(units, tasks)


def test_publish_existing_uses_campaign_task_and_finalize_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = _write_unit(tmp_path / "unit")

    class RecordingClient:
        instance: "RecordingClient"

        def __init__(self, base_url: str, token: str) -> None:
            assert base_url == "https://scoreboard.test"
            assert token == "secret"
            self.campaign: dict[str, object] | None = None
            self.tasks: list[tuple[str, str, dict[str, object]]] = []
            self.finalized: tuple[str, int] | None = None
            RecordingClient.instance = self

        def preflight(
            self,
            evaluator: str,
            version: str,
            campaign_schema: str | None = None,
        ) -> dict[str, object]:
            assert (evaluator, version) == ("lm-eval", "0.4.12")
            assert campaign_schema == existing.EXISTING_CAMPAIGN_SCHEMA
            return {"status": "ready"}

        def create_campaign(
            self,
            payload: dict[str, object],
            run_key: str,
        ) -> dict[str, object]:
            assert run_key == payload["run_key"]
            self.campaign = payload
            return {"campaign_id": CAMPAIGN_ID}

        def publish_task(
            self,
            campaign_id: str,
            task_identity: str,
            payload: dict[str, object],
        ) -> None:
            self.tasks.append((campaign_id, task_identity, payload))

        def finalize(self, campaign_id: str, expected_count: int) -> None:
            self.finalized = (campaign_id, expected_count)

    monkeypatch.setattr(existing, "ScoreboardClient", RecordingClient)

    result = existing.publish_existing(
        output_dirs=[output_dir],
        env={
            "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.test",
            "HELICOPTER_SCOREBOARD_TOKEN": "secret",
        },
        dry_run=False,
    )

    client = RecordingClient.instance
    assert result == 0
    assert client.campaign is not None
    assert client.campaign["schema_version"] == existing.EXISTING_CAMPAIGN_SCHEMA
    assert len(client.tasks) == 1
    assert client.tasks[0][0] == CAMPAIGN_ID
    assert client.tasks[0][2]["details"][0]["doc"]["question"] == "2 + 2?"
    assert client.finalized == (CAMPAIGN_ID, 1)


def test_publish_existing_dry_run_preflights_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = _write_unit(tmp_path / "unit")

    class DryRunClient:
        def __init__(self, _base_url: str, _token: str) -> None:
            pass

        def preflight(
            self,
            _evaluator: str,
            _version: str,
            campaign_schema: str | None = None,
        ) -> dict[str, object]:
            return {"status": "ready", "schema": campaign_schema}

        def create_campaign(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("dry-run must not create a campaign")

    monkeypatch.setattr(existing, "ScoreboardClient", DryRunClient)

    result = existing.publish_existing(
        output_dirs=[output_dir],
        env={
            "HELICOPTER_SCOREBOARD_URL": "https://scoreboard.test",
            "HELICOPTER_SCOREBOARD_TOKEN": "secret",
        },
        dry_run=True,
    )

    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["mode"] == "publish-existing"
    assert report["expected_task_count"] == 1
    assert report["execution_units"][0]["summary_sha256"]


def test_lm_eval_entrypoint_reports_existing_artifact_errors_cleanly(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="existing output directory is invalid"):
        entrypoint.main(
            [
                "--publish-existing",
                "--output-dir",
                str(tmp_path / "missing"),
                "--dry-run",
            ]
        )


def test_lm_eval_entrypoint_rejects_publication_options_for_eval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        entrypoint.main(["--config", "eval.toml", "--weight-sha256", "a" * 64])

    assert raised.value.code == 2
    assert "publication options require --publish-existing" in capsys.readouterr().err
