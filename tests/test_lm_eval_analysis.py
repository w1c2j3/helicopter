from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from helicopter_lm_eval.analysis import (
    analyze_samples,
    build_task_records,
    main,
    render_markdown,
    render_task_markdown,
)
from helicopter_lm_eval.config import LMEvalConfig
from helicopter_lm_eval.evaluate import _write_results


def test_analysis_recovers_choice_answer_and_score_margin() -> None:
    samples = {
        "race": [
            {
                "doc_id": 7,
                "doc": {"article": "context"},
                "arguments": [
                    ["Question: Which answer?", " first"],
                    ["Question: Which answer?", " second"],
                ],
                "filtered_resps": [[-8.0, False], [-2.5, False]],
                "target": 0,
                "acc": 0.0,
            },
            {
                "doc_id": 8,
                "arguments": [["prompt", " yes"], ["prompt", " no"]],
                "filtered_resps": [[-1.0, False], [-3.0, False]],
                "target": 0,
                "acc": True,
            },
        ]
    }

    analysis, bad_cases = analyze_samples(samples)

    assert analysis["scored"] == 2
    assert analysis["incorrect"] == 1
    case = bad_cases["cases"][0]
    assert case["error_type"] == "wrong_choice"
    assert case["model_answer"] == " second"
    assert case["standard_answer"] == " first"
    assert case["score_margin_over_target"] == 5.5
    assert case["doc_id"] == 7


def test_task_records_include_every_model_answer_and_wrong_reason() -> None:
    samples = [
        {
            "doc_id": 1,
            "doc": {
                "Question": "Which material is essential?",
                "A": "Air",
                "B": "Land",
            },
            "arguments": [["prompt", " A"], ["prompt", " B"]],
            "filtered_resps": [[-1.0, False], [-3.0, False]],
            "target": 1,
            "acc": 0,
            "acc_norm": 1,
        },
        {
            "doc_id": 2,
            "arguments": [["prompt", " yes"], ["prompt", " no"]],
            "filtered_resps": [[-1.0, False], [-3.0, False]],
            "target": 0,
            "acc": 1,
        },
    ]

    records = build_task_records("cmmlu_agronomy", samples)

    assert len(records) == 2
    assert records[0]["status"] == "incorrect"
    assert records[0]["question"] == "Which material is essential?"
    assert records[0]["model_answer"] == "A. Air"
    assert records[0]["standard_answer"] == "B. Land"
    assert records[0]["binary_metrics"] == {"acc": 0.0, "acc_norm": 1.0}
    assert "分数优势" in records[0]["why_wrong"]
    assert records[1]["status"] == "correct"


def test_filter_variants_collapse_to_one_record_with_raw_output() -> None:
    raw_output = "Reasoning through the problem. The final answer is 18."
    common = {
        "doc_id": 0,
        "doc_hash": "doc-hash",
        "prompt_hash": "prompt-hash",
        "target_hash": "target-hash",
        "doc": {"question": "How much money did Janet make?"},
        "resps": [[raw_output]],
        "target": ["18"],
    }
    samples = [
        {
            **common,
            "filter": "strict-match",
            "filtered_resps": ["[invalid]"],
            "exact_match": 0,
        },
        {
            **common,
            "filter": "flexible-extract",
            "filtered_resps": ["18"],
            "exact_match": 0,
        },
    ]

    records = build_task_records("gsm8k", samples)
    analysis, _bad_cases = analyze_samples({"gsm8k": samples})

    assert len(records) == 1
    assert records[0]["status"] == "incorrect"
    assert records[0]["filter"] == "flexible-extract"
    assert records[0]["model_output"] == raw_output
    assert records[0]["model_answer"] == "18"
    assert records[0]["filter_results"] == {
        "strict-match": {
            "binary_metrics": {"exact_match": 0.0},
            "filtered_response": "[invalid]",
        },
        "flexible-extract": {
            "binary_metrics": {"exact_match": 0.0},
            "filtered_response": "18",
        },
    }
    assert analysis["samples"] == 1
    assert analysis["incorrect"] == 1
    markdown = render_task_markdown(analysis["tasks"][0], records)
    assert "模型原始输出" in markdown
    assert raw_output in markdown


def test_analysis_distinguishes_retrieval_format_and_location_errors() -> None:
    samples = {
        "longbench_passage_retrieval_en": [
            {
                "doc_id": 1,
                "doc": {"question": "Find the source paragraph."},
                "filtered_resps": [" Paragraph 1"],
                "target": ["Paragraph 15"],
                "retrieval_score": 0.0,
            },
            {
                "doc_id": 2,
                "doc": {"question": "Find the source paragraph."},
                "filtered_resps": [" I cannot tell"],
                "target": ["Paragraph 4"],
                "retrieval_score": 0.0,
            },
        ]
    }

    _analysis, bad_cases = analyze_samples(samples)

    assert [case["error_type"] for case in bad_cases["cases"]] == [
        "wrong_paragraph",
        "retrieval_format_error",
    ]
    assert bad_cases["cases"][0]["predicted_paragraph"] == 1
    assert bad_cases["cases"][0]["expected_paragraph"] == 15


def test_continuous_generation_metric_is_diagnostic_not_incorrect() -> None:
    samples = {
        "wmt14-en-fr": [
            {
                "doc_id": 3,
                "doc": {
                    "translation": {"en": "A cat", "fr": "Un chat"}
                },
                "filtered_resps": [" 999999999999"],
                "target": " Un chat",
                "bleu": [" Un chat", " 999999999999"],
            }
        ]
    }

    analysis, bad_cases = analyze_samples(samples)

    assert analysis["incorrect"] == 0
    assert analysis["unscored"] == 1
    assert bad_cases["cases"][0]["kind"] == "quality_outlier"
    assert bad_cases["cases"][0]["error_type"] == "low_reference_overlap"
    assert "Diagnostic only" in bad_cases["cases"][0]["note"]


def test_likelihood_failure_does_not_invent_model_answer() -> None:
    samples = {
        "lambada_openai": [
            {
                "doc_id": 11,
                "arguments": [["The final", " word"]],
                "filtered_resps": [[-4.25, False]],
                "target": " word",
                "acc": 0,
            }
        ]
    }

    analysis, bad_cases = analyze_samples(samples)

    assert analysis["incorrect"] == 1
    case = bad_cases["cases"][0]
    assert case["error_type"] == "target_not_greedy"
    assert case["model_answer"] is None
    assert case["scored_target_loglikelihood"] == -4.25


def test_markdown_bounds_representative_cases_per_family() -> None:
    samples = {
        "blimp_one": [
            {
                "doc_id": index,
                "arguments": [["", " good"], ["", " bad"]],
                "filtered_resps": [[-2.0, False], [-1.0, False]],
                "target": 0,
                "acc": 0,
            }
            for index in range(8)
        ]
    }
    analysis, bad_cases = analyze_samples(samples, examples_per_task=8)

    markdown = render_markdown(analysis, bad_cases)

    assert markdown.count("### blimp_one") == 3


def test_generation_diagnostic_detects_repetition_loop() -> None:
    repeated = " je suis un homme" * 8
    samples = {
        "wmt14-en-fr": [
            {
                "doc_id": 4,
                "filtered_resps": [repeated],
                "target": " une traduction normale",
            }
        ]
    }

    analysis, bad_cases = analyze_samples(samples)

    assert analysis["tasks"][0]["diagnostic_types"] == {"repetition_loop": 1}
    assert bad_cases["cases"][0]["error_type"] == "repetition_loop"


def test_result_writer_registers_error_analysis_artifacts(tmp_path: Path) -> None:
    config = LMEvalConfig(
        tasks=("race",),
        output_dir=tmp_path,
        batch_size=1,
        eot_token_id=0,
        max_gen_toks=32,
        limit=1,
        log_samples=True,
        vllm_pool_manifest=tmp_path / "pool.json",
        manifest=SimpleNamespace(
            global_step=1,
            wkv_mode="fp16",
            max_model_len=128,
        ),
    )
    raw = {
        "results": {"race": {"acc,none": 0.0}},
        "samples": {
            "race": [
                {
                    "doc_id": 0,
                    "arguments": [["prompt", " A"], ["prompt", " B"]],
                    "filtered_resps": [[-2.0, False], [-1.0, False]],
                    "target": 0,
                    "acc": 0,
                }
            ]
        },
        "versions": {"race": 1},
    }

    _write_results(
        config=config,
        model_id="rwkv-test",
        version="0.4.12",
        results=raw,
    )

    artifacts = json.loads((tmp_path / "artifacts.json").read_text())
    assert artifacts["error_analysis_path"] == "error_analysis.json"
    assert artifacts["bad_cases_path"] == "bad_cases.json"
    assert artifacts["error_analysis_markdown_path"] == "error_analysis.md"
    assert json.loads((tmp_path / "error_analysis.json").read_text())[
        "incorrect"
    ] == 1
    benchmark = tmp_path / "benchmarks" / "race"
    records = [
        json.loads(line)
        for line in (benchmark / "records.jsonl").read_text().splitlines()
    ]
    errors = [
        json.loads(line)
        for line in (benchmark / "errors.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert errors == records
    assert "模型答案" in (benchmark / "report.md").read_text()
    assert artifacts["benchmark_artifacts"][0]["task_name"] == "race"


def test_posthoc_analysis_preserves_original_artifact_manifest(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "samples": {
                    "race": [
                        {
                            "doc_id": 0,
                            "arguments": [
                                ["prompt", " A"],
                                ["prompt", " B"],
                            ],
                            "filtered_resps": [
                                [-2.0, False],
                                [-1.0, False],
                            ],
                            "target": 0,
                            "acc": 0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text('{"schema_version": 1}\n', encoding="utf-8")

    assert main(["--results", str(results)]) == 0

    assert json.loads(artifacts.read_text()) == {"schema_version": 1}
    analysis_artifacts = json.loads(
        (tmp_path / "analysis_artifacts.json").read_text()
    )
    assert analysis_artifacts["source_results_path"] == "results.json"
    assert len(analysis_artifacts["source_results_sha256"]) == 64
    assert analysis_artifacts["benchmark_artifacts"][0]["task_name"] == "race"
    assert (tmp_path / "benchmarks" / "race" / "report.md").is_file()
