from __future__ import annotations

import os


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
