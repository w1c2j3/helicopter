from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any

import requests
from tqdm import tqdm

from helicopter_cli.lighteval_raw_completion import _configured_prompt_template, _render_prompt


def _completion_url(base_url: str | None) -> str:
    if not base_url:
        raise RuntimeError("LiteLLM base_url is required for OpenAI-compatible logprob patch")
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/completions"
    return f"{base}/v1/completions"


def _served_model_name(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _choice_token_logprobs(payload: dict[str, Any], *, context_chars: int, prompt_chars: int) -> tuple[float, bool, list[str]]:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"completion logprob response has no choices: {payload!r}")
    logprobs = choices[0].get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    offsets = logprobs.get("text_offset") or []
    top_logprobs = logprobs.get("top_logprobs") or []
    if not (len(tokens) == len(token_logprobs) == len(offsets)):
        raise RuntimeError("completion logprob response has inconsistent token/logprob/offset lengths")

    selected_logprobs: list[float] = []
    selected_tokens: list[str] = []
    selected_argmax: list[bool] = []
    next_offsets = list(offsets[1:]) + [prompt_chars]
    for token, logprob, offset, next_offset, top in zip(tokens, token_logprobs, offsets, next_offsets, top_logprobs):
        if logprob is None:
            continue
        if next_offset <= context_chars or offset >= prompt_chars:
            continue
        selected_logprobs.append(float(logprob))
        selected_tokens.append(str(token))
        if isinstance(top, dict) and top:
            best_token = max(top.items(), key=lambda item: item[1])[0]
            selected_argmax.append(str(best_token) == str(token))
        else:
            selected_argmax.append(False)
    if not selected_logprobs:
        raise RuntimeError("completion logprob response did not include continuation token logprobs")
    return sum(selected_logprobs), all(selected_argmax), selected_tokens


def patch_litellm_logprobs() -> None:
    from lighteval.data import LoglikelihoodDataset
    from lighteval.models.endpoints.litellm_model import LiteLLMClient
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import SamplingMethod
    from lighteval.utils.cache_management import cached

    response_fields = {
        "input",
        "input_tokens",
        "text",
        "output_tokens",
        "text_post_processed",
        "reasonings",
        "logprobs",
        "argmax_logits_eq_gold",
        "logits",
        "unconditioned_logprobs",
        "truncated_tokens_count",
        "padded_tokens_count",
    }

    def restore_response(payload: dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            **{key: value for key, value in payload.items() if key in response_fields}
        )

    def stored_loglikelihood(task_id: str | None) -> dict[int, dict[str, Any]]:
        if not task_id:
            return {}
        from helicopter_cli.scoreboard_bridge import load_lighteval_generation

        stored: dict[int, dict[str, Any]] = {}
        for row in load_lighteval_generation(task_id=task_id):
            if str(row.get("status") or "").lower() != "completed":
                continue
            context = row.get("context")
            if not isinstance(context, dict):
                continue
            stats = context.get("stats")
            if not isinstance(stats, dict) or not stats.get("loglikelihood_checkpoint"):
                continue
            agent_result = context.get("agent_result")
            if not isinstance(agent_result, dict):
                continue
            response = agent_result.get("model_response")
            doc = agent_result.get("doc")
            if not isinstance(response, dict) or not isinstance(doc, dict):
                continue
            stored[int(row["sample_index"])] = {
                "model_response": response,
                "dataset_row_id": doc.get("id"),
                "prompt": response.get("input"),
            }
        return stored

    def _call_completion_logprobs(self: LiteLLMClient, prompt: str) -> dict[str, Any]:
        url = _completion_url(self.base_url)
        headers = {"Content-Type": "application/json"}
        # The endpoint command exports the model credential through the
        # environment instead of including it in LiteLLM's model args.  Keep
        # logprob requests on the same auth path as raw completions.
        api_key = (
            os.environ.get("HELICOPTER_EVAL_API_KEY")
            or self.api_key
            or os.environ.get("OPENAI_API_KEY")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": _served_model_name(self.model),
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1e-5,
            "echo": True,
            "logprobs": 1,
        }
        last_error: Exception | None = None
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout or 120)
                response.raise_for_status()
                return response.json()
            except Exception as error:  # noqa: BLE001 - mirror LiteLLM retry behavior
                last_error = error
                wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                time.sleep(wait_time)
        raise RuntimeError(f"completion logprob request failed after retries: {last_error}")

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self: LiteLLMClient, docs: list[Any]) -> list[Any]:
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        default_template = os.environ.get("HELICOPTER_PROMPT_TEMPLATE", "")
        task_request_policy = os.environ.get(
            "HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY",
            os.environ.get("HELICOPTER_LIGHTEEVAL_TASK_REQUEST_POLICY", ""),
        )
        task_id = os.environ.get("HELICOPTER_SCOREBOARD_TASK_ID", "").strip() or None
        dataset_name = os.environ.get("HELICOPTER_SCOREBOARD_DATASET", "").strip()
        stored = stored_loglikelihood(task_id)
        original_indices = {id(doc): index for index, doc in enumerate(docs)}
        results: list[Any] = []

        if task_id or dataset_name:
            from helicopter_cli.scoreboard_bridge import LightevalCheckpointSession

            checkpoint_context = LightevalCheckpointSession(
                task_id=task_id,
                dataset=dataset_name,
                num_samples=len(docs),
            )
        else:
            checkpoint_context = nullcontext(None)

        with checkpoint_context as checkpoint_session:
            for split in tqdm(
                dataset.splits_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=self.disable_tqdm,
            ):
                split_docs = list(split)
                contexts: list[str] = []
                task_names: list[str] = []
                sample_indices: list[int] = []
                for doc in split_docs:
                    sample_index = original_indices.get(id(doc))
                    if sample_index is None:
                        raise RuntimeError(
                            "LightEval changed document identity; cannot assign a stable sample_index"
                        )
                    task_name = str(getattr(doc, "task_name", "") or dataset_name)
                    task_names.append(task_name)
                    sample_indices.append(sample_index)
                    if default_template or task_request_policy:
                        template = _configured_prompt_template(task_name, default_template)
                        contexts.append(
                            _render_prompt(
                                template,
                                str(getattr(doc, "query", "")),
                                task_name=task_name,
                            )
                        )
                    else:
                        contexts.append(str(getattr(doc, "query", "")))

                responses: list[ModelResponse | None] = [None] * len(split_docs)
                grouped_logprobs: list[list[float | None]] = [
                    [None] * len(doc.choices) for doc in split_docs
                ]
                grouped_argmax: list[list[bool | None]] = [
                    [None] * len(doc.choices) for doc in split_docs
                ]
                grouped_tokens: list[list[list[str] | None]] = [
                    [None] * len(doc.choices) for doc in split_docs
                ]
                jobs: list[tuple[int, int, str, str]] = []
                for doc_index, (doc, context, sample_index) in enumerate(
                    zip(split_docs, contexts, sample_indices)
                ):
                    checkpoint = stored.get(sample_index)
                    if checkpoint is not None:
                        stored_doc_id = checkpoint.get("dataset_row_id")
                        current_doc_id = getattr(doc, "id", None)
                        if stored_doc_id is None or (
                            current_doc_id is not None
                            and str(stored_doc_id) != str(current_doc_id)
                        ):
                            raise RuntimeError(
                                f"loglikelihood checkpoint {sample_index} belongs to a different dataset row"
                            )
                        if checkpoint.get("prompt") != context:
                            raise RuntimeError(
                                f"loglikelihood checkpoint {sample_index} prompt changed; refusing resume"
                            )
                        responses[doc_index] = restore_response(checkpoint["model_response"])
                        continue
                    for choice_index, choice in enumerate(doc.choices):
                        full_prompt = context + str(choice)
                        jobs.append((doc_index, choice_index, context, full_prompt))

                def finish_document(doc_index: int) -> None:
                    logprobs = grouped_logprobs[doc_index]
                    argmax = grouped_argmax[doc_index]
                    tokens = grouped_tokens[doc_index]
                    if any(value is None for value in (*logprobs, *argmax, *tokens)):
                        return
                    response = ModelResponse(
                        input=contexts[doc_index],
                        logprobs=[float(value) for value in logprobs if value is not None],
                        argmax_logits_eq_gold=[bool(value) for value in argmax if value is not None],
                        output_tokens=[value for value in tokens if value is not None],
                    )
                    if checkpoint_session is not None:
                        checkpoint_session.checkpoint_loglikelihood(
                            task_name=task_names[doc_index],
                            sample_index=sample_indices[doc_index],
                            doc=split_docs[doc_index],
                            response=response,
                        )
                    responses[doc_index] = response

                with ThreadPoolExecutor(self.concurrent_requests) as executor:
                    futures = {
                        executor.submit(_call_completion_logprobs, self, full_prompt): (
                            doc_index,
                            choice_index,
                            context,
                            full_prompt,
                        )
                        for doc_index, choice_index, context, full_prompt in jobs
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Loglikelihoods",
                        position=1,
                        leave=False,
                        disable=self.disable_tqdm,
                    ):
                        doc_index, choice_index, context, full_prompt = futures[future]
                        response_payload = future.result()
                        logprob, argmax, tokens = _choice_token_logprobs(
                            response_payload,
                            context_chars=len(context),
                            prompt_chars=len(full_prompt),
                        )
                        grouped_logprobs[doc_index][choice_index] = logprob
                        grouped_argmax[doc_index][choice_index] = argmax
                        grouped_tokens[doc_index][choice_index] = tokens
                        finish_document(doc_index)

                for response in responses:
                    if response is None:
                        raise RuntimeError("loglikelihood returned an incomplete response set")
                    results.append(response)

        return dataset.get_original_order(results)

    LiteLLMClient.loglikelihood = loglikelihood


patch_litellm_logprobs()
