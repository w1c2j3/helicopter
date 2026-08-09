from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_evalscope_agent_matrix.py"
_SPEC = importlib.util.spec_from_file_location("summarize_evalscope_agent_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_matrix = _MODULE.summarize_matrix


def _write_report(path: Path, *, exit_code: int, sample_count: int, official_count: int) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "samples": [{} for _ in range(sample_count)],
                "counts": {"correct": sample_count},
                "official_reports": [
                    {
                        "report": {
                            "dataset_name": "bfcl_v3",
                            "model_name": "rwkv-test",
                            "score": 0.5,
                            "num": official_count,
                            "metrics": [{"name": "accuracy", "score": 0.5}],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_summary_only_scores_complete_matching_reports(tmp_path: Path) -> None:
    _write_report(tmp_path / "complete" / "raw" / "acceptance_report.json", exit_code=0, sample_count=3, official_count=3)
    _write_report(tmp_path / "partial" / "raw" / "acceptance_report.json", exit_code=0, sample_count=3, official_count=2)
    _write_report(tmp_path / "failed" / "raw" / "acceptance_report.json", exit_code=1, sample_count=3, official_count=3)

    summary = summarize_matrix([tmp_path])

    assert len(summary["complete"]) == 1
    assert summary["complete"][0]["score"] == 0.5
    assert len(summary["incomplete"]) == 2
