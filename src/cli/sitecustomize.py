from __future__ import annotations

import os


if os.environ.get("HELICOPTER_LIGHTEEVAL_ASSERT_LOCAL_SOURCE") == "1":
    from pathlib import Path

    import lighteval

    expected_root = Path(os.environ["HELICOPTER_LIGHTEEVAL_SOURCE_ROOT"]).resolve()
    actual_file_name = getattr(lighteval, "__file__", None)
    if not actual_file_name:
        raise ImportError("LightEval resolved as a namespace package; local source was not loaded")
    actual_file = Path(actual_file_name).resolve()
    try:
        actual_file.relative_to(expected_root)
    except ValueError as error:
        raise ImportError(
            "LightEval is not using the repository source: "
            f"{actual_file} (expected under {expected_root})"
        ) from error
    print(f"helicopter: using local LightEval source {actual_file}", flush=True)


if os.environ.get("HELICOPTER_PATCH_LIGHTEVAL_LITELLM_LOGPROBS") == "1":
    import helicopter_cli.lighteval_litellm_logprobs  # noqa: F401

if os.environ.get("HELICOPTER_VLLM_SAMPLING_JSON"):
    import helicopter_cli.lighteval_vllm_sampling  # noqa: F401

if os.environ.get("HELICOPTER_PATCH_LIGHTEVAL_DATASET_RETRIES") == "1":
    import helicopter_cli.lighteval_dataset_resilience  # noqa: F401
if os.environ.get("HELICOPTER_PROMPT_TEMPLATE"):
    import helicopter_cli.lighteval_raw_completion  # noqa: F401

if os.environ.get("HELICOPTER_SCOREBOARD_DB_ONLY") == "1":
    import helicopter_cli.lighteval_scoreboard  # noqa: F401
    import helicopter_cli.lighteval_db_pipeline  # noqa: F401
