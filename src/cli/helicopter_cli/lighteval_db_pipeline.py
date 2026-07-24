from __future__ import annotations

import json
import logging
import os
from typing import Any

from lighteval.pipeline import Pipeline
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.requests import SamplingMethod

from helicopter_cli.lighteval_answer_adapters import adapt_answer
from helicopter_cli.lighteval_raw_completion import _doc_id, _identity_value, _response_from_rollouts
from helicopter_cli.scoreboard_bridge import load_lighteval_generation


_ORIGINAL_EVALUATE = Pipeline.evaluate
_ORIGINAL_SHOW_RESULTS = Pipeline.show_results
_ORIGINAL_GET_RESULTS = Pipeline.get_results
_ORIGINAL_SAVE_RESULTS = Pipeline.save_and_push_results
_ORIGINAL_POST_PROCESS_OUTPUTS = Pipeline._post_process_outputs
_ORIGINAL_GET_DOCS = LightevalTask.get_docs
logger = logging.getLogger(__name__)


def auto_sample_count(
    *,
    document_count: int,
    rollout_n: int,
    target_generations: int,
    large_generation_threshold: int,
    large_sample_rate: float,
) -> int | None:
    """Return the configured deterministic sample cap for one benchmark."""

    document_count = max(0, int(document_count))
    rollout_n = max(1, int(rollout_n))
    full_generations = document_count * rollout_n
    if not document_count or full_generations <= int(target_generations):
        return None
    if full_generations >= int(large_generation_threshold):
        selected = round(document_count * float(large_sample_rate))
    else:
        selected = round(int(target_generations) / rollout_n)
    return max(1, min(document_count, int(selected)))


def strip_prefilled_reasoning(text: str, *, force: bool = False) -> str:
    """Return the answer after a prefilled think block closes.

    The prompt owns the opening think text, so the generated continuation
    contains the closing tag. When the model quotes a literal closing tag from
    the prompt before closing its real reasoning block, the final closing tag
    is the only valid answer boundary. Full in-response tag pairs remain
    delegated to LightEval's native reasoning-tag remover.
    """

    value = str(text or "")
    lowered = value.lower()
    closing = lowered.rfind("</think>") if force else lowered.find("</think>")
    opening = lowered.find("<think")
    if closing >= 0 and (force or opening < 0 or opening > closing):
        return value[closing + len("</think>") :].lstrip()
    return value


def has_unclosed_reasoning_prefill(prompt: Any) -> bool:
    value = str(prompt or "").lower()
    opening = value.rfind("<think")
    closing = value.rfind("</think>")
    return opening >= 0 and opening > closing


def has_empty_reasoning_prefill(prompt: Any) -> bool:
    """Return whether the prompt ends with the configured ``</think`` prefill.

    The missing final ``>`` is deliberate: the model emits it as the first
    continuation token and the request adapter removes that token before the
    response reaches LightEval.  Consequently the whole remaining continuation
    is an answer and must not be passed through reasoning-tag removal.
    """

    value = str(prompt or "").rstrip().lower()
    return value.endswith("</think") and not value.endswith("</think>")


def _configured_sample_policy() -> dict[str, Any] | None:
    raw = os.environ.get("HELICOPTER_LIGHTEEVAL_G1H_POLICY", "").strip()
    if not raw:
        return None
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError:
        return None
    required = {
        "target_generations_per_benchmark",
        "large_benchmark_generation_threshold",
        "large_benchmark_sample_rate",
    }
    return policy if isinstance(policy, dict) and required <= policy.keys() else None


def _configured_request_policy() -> dict[str, Any] | None:
    raw = os.environ.get("HELICOPTER_LIGHTEVAL_TASK_REQUEST_POLICY", "").strip()
    if not raw:
        return None
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError:
        return None
    tasks = policy.get("tasks") if isinstance(policy, dict) else None
    if not isinstance(tasks, dict) or len(tasks) != 1:
        return None
    entry = next(iter(tasks.values()))
    if not isinstance(entry, dict):
        return None
    return entry if isinstance(entry, dict) else None


def _configured_request_format() -> str | None:
    entry = _configured_request_policy()
    if entry is None:
        return None
    value = entry.get("format")
    return str(value).strip() or None if value is not None else None


def normalize_code_fences(text: str) -> str:
    """Give LightEval's official code extractor exactly one fenced block.

    The official LCB extractor uses the final two fence lines and returns an
    empty program when fewer than two exist. Preserve that exact extraction
    choice while closing a truncated single block and removing unrelated
    earlier blocks from the scoring text.
    """

    value = str(text or "")
    lines = value.splitlines()
    fence_lines = [index for index, line in enumerate(lines) if "```" in line]
    if not fence_lines:
        return value
    if len(fence_lines) == 2:
        return value
    if len(fence_lines) == 1:
        return value.rstrip() + "\n```"

    opening = lines[fence_lines[-2]]
    language = "python" if "python" in opening.casefold() else ""
    body = "\n".join(lines[fence_lines[-2] + 1 : fence_lines[-1]])
    return f"```{language}\n{body.rstrip()}\n```"


def _get_docs(self: LightevalTask, max_samples: int | None = None) -> list[Any]:
    policy = _configured_sample_policy()
    if max_samples is not None or policy is None:
        return _ORIGINAL_GET_DOCS(self, max_samples)
    raw_num_samples = getattr(self, "num_samples", [1])
    rollout_n = (
        max(raw_num_samples)
        if isinstance(raw_num_samples, (list, tuple))
        else int(raw_num_samples)
    )
    document_count = len(self.eval_docs())
    selected = auto_sample_count(
        document_count=document_count,
        rollout_n=rollout_n,
        target_generations=int(policy["target_generations_per_benchmark"]),
        large_generation_threshold=int(policy["large_benchmark_generation_threshold"]),
        large_sample_rate=float(policy["large_benchmark_sample_rate"]),
    )
    if selected is not None:
        logger.info(
            "benchmark sampling: task=%s documents=%d selected=%d rollout_n=%d",
            self.name,
            document_count,
            selected,
            rollout_n,
        )
    return _ORIGINAL_GET_DOCS(self, selected)


def _stage() -> str:
    return os.environ.get("HELICOPTER_PIPELINE_STAGE", "full").strip().lower()


def _responses_from_database(pipeline: Pipeline) -> dict[Any, list[Any]]:
    task_id = os.environ.get("HELICOPTER_SCOREBOARD_TASK_ID", "").strip()
    if not task_id:
        raise RuntimeError("score stage requires HELICOPTER_SCOREBOARD_TASK_ID")
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for row in load_lighteval_generation(task_id=task_id):
        if str(row.get("status") or "").lower() != "completed":
            continue
        context = row.get("context")
        agent_result = context.get("agent_result") if isinstance(context, dict) else None
        response = agent_result.get("model_response") if isinstance(agent_result, dict) else None
        if not isinstance(response, dict):
            continue
        stats = context.get("stats") if isinstance(context.get("stats"), dict) else {}
        stored_doc = agent_result.get("doc") if isinstance(agent_result.get("doc"), dict) else {}
        grouped.setdefault(int(row["sample_index"]), {})[int(row["repeat_index"])] = {
            "model_response": response,
            "dataset_row_id": stats.get(
                "dataset_row_id",
                stats.get("lighteval_doc_id", stored_doc.get("id", context.get("task_id"))),
            ),
        }

    outputs: dict[Any, list[Any]] = {}
    for sampling_method, docs in pipeline.sampling_docs.items():
        if sampling_method == SamplingMethod.LOGPROBS:
            from lighteval.models.model_output import ModelResponse

            responses = []
            for sample_index, doc in enumerate(docs):
                stored = grouped.get(sample_index, {})
                checkpoint = stored.get(0)
                if checkpoint is None:
                    raise RuntimeError(
                        f"task {task_id} generation is incomplete at loglikelihood sample "
                        f"{sample_index}: missing response"
                    )
                stored_doc_id = checkpoint.get("dataset_row_id")
                current_doc_id = _doc_id(doc)
                if stored_doc_id is None:
                    raise RuntimeError(
                        f"task {task_id} loglikelihood checkpoint {sample_index} has no dataset identity"
                    )
                if (
                    current_doc_id is not None
                    and _identity_value(stored_doc_id) != _identity_value(current_doc_id)
                ):
                    raise RuntimeError(
                        f"task {task_id} loglikelihood checkpoint {sample_index} belongs to dataset row "
                        f"{stored_doc_id!r}, not current row {current_doc_id!r}"
                    )
                response_payload = checkpoint["model_response"]
                if not isinstance(response_payload, dict):
                    raise RuntimeError(
                        f"task {task_id} loglikelihood checkpoint {sample_index} has no model response"
                    )
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
                responses.append(
                    ModelResponse(
                        **{
                            key: value
                            for key, value in response_payload.items()
                            if key in response_fields
                        }
                    )
                )
            outputs[sampling_method] = responses
            continue
        if sampling_method != SamplingMethod.GENERATIVE:
            raise RuntimeError(f"database score stage does not support {sampling_method}")
        responses = []
        for sample_index, doc in enumerate(docs):
            rollout_n = max(1, int(getattr(doc, "num_samples", 1)))
            stored = grouped.get(sample_index, {})
            missing = [index for index in range(rollout_n) if index not in stored]
            if missing:
                raise RuntimeError(
                    f"task {task_id} generation is incomplete at sample {sample_index}: missing repeats {missing}"
                )
            current_doc_id = _doc_id(doc)
            rollouts = []
            for repeat_index in range(rollout_n):
                checkpoint = stored[repeat_index]
                stored_doc_id = checkpoint.get("dataset_row_id")
                if stored_doc_id is None:
                    raise RuntimeError(
                        f"task {task_id} checkpoint {sample_index}/{repeat_index} has no dataset "
                        "identity; refusing unsafe index-only scoring"
                    )
                if (
                    current_doc_id is not None
                    and _identity_value(stored_doc_id) != _identity_value(current_doc_id)
                ):
                    raise RuntimeError(
                        f"task {task_id} checkpoint {sample_index}/{repeat_index} belongs to dataset "
                        f"row {stored_doc_id!r}, not current row {current_doc_id!r}; refusing "
                        "mismatched scoring"
                    )
                rollouts.append(checkpoint["model_response"])
            responses.append(
                _response_from_rollouts(rollouts)
            )
        outputs[sampling_method] = responses
    return outputs


def _post_process_outputs(
    self: Pipeline,
    sampling_method_responses: dict[str, list[Any]],
) -> None:
    _ORIGINAL_POST_PROCESS_OUTPUTS(self, sampling_method_responses)
    policy = _configured_request_policy() or {}
    domain = policy.get("domain")
    request_format = policy.get("format")
    configured_stops = policy.get("stop")
    stops = configured_stops if isinstance(configured_stops, list) else []
    for responses in sampling_method_responses.values():
        for response in responses:
            texts = list(getattr(response, "text", None) or [])
            processed = list(getattr(response, "text_post_processed", None) or texts)
            prompt = str(getattr(response, "input", None) or "")
            requires_closing = has_unclosed_reasoning_prefill(prompt)
            empty_reasoning_prefill = has_empty_reasoning_prefill(prompt)
            scored: list[str] = []
            for index, text in enumerate(texts):
                raw = str(text or "")
                candidate = processed[index] if index < len(processed) else raw
                # Native LightEval may set text_post_processed to an empty
                # string when the prompt itself owns the reasoning prefill.
                # That is not an answer; keep the raw completion available to
                # the RWKV adapter and to the task metric.
                if not str(candidate or "").strip():
                    candidate = raw
                if empty_reasoning_prefill:
                    scored.append(raw)
                elif requires_closing and "</think>" not in raw.lower():
                    scored.append(str(candidate))
                elif requires_closing:
                    scored.append(strip_prefilled_reasoning(raw, force=True))
                elif self.pipeline_parameters.remove_reasoning_tags:
                    scored.append(strip_prefilled_reasoning(str(candidate)))
                else:
                    scored.append(str(candidate))
            response.text_post_processed = [
                adapt_answer(
                    value,
                    domain=domain,
                    request_format=request_format,
                    prompt=prompt,
                    stops=stops,
                )
                for value in scored
            ]


def _evaluate(self: Pipeline) -> None:
    stage = _stage()
    if stage not in {"generate", "score"}:
        return _ORIGINAL_EVALUATE(self)

    self.evaluation_tracker.general_config_logger.log_args_info(
        num_fewshot_seeds=self.pipeline_parameters.num_fewshot_seeds,
        max_samples=self.pipeline_parameters.max_samples,
        job_id=str(self.pipeline_parameters.job_id),
    )
    if stage == "generate":
        self._run_model()
        self.evaluation_tracker.general_config_logger.log_end_time()
        self._helicopter_generation_only = True
        return

    outputs = _responses_from_database(self)
    if self.is_main_process():
        self._post_process_outputs(outputs)
        self._compute_metrics(outputs)
        self.evaluation_tracker.general_config_logger.log_end_time()
        self.evaluation_tracker.metrics_logger.aggregate(
            task_dict=self.tasks_dict,
            bootstrap_iters=self.pipeline_parameters.bootstrap_iters,
        )
        self.evaluation_tracker.details_logger.aggregate()


def _show_results(self: Pipeline) -> None:
    if getattr(self, "_helicopter_generation_only", False):
        print("lighteval: generation complete; scoring deferred to database worker")
        return
    _ORIGINAL_SHOW_RESULTS(self)


def _get_results(self: Pipeline) -> Any:
    if getattr(self, "_helicopter_generation_only", False):
        return {"generation_only": True}
    return _ORIGINAL_GET_RESULTS(self)


def _save_and_push_results(self: Pipeline) -> None:
    if getattr(self, "_helicopter_generation_only", False):
        return
    _ORIGINAL_SAVE_RESULTS(self)


if not getattr(Pipeline.evaluate, "_helicopter_db_pipeline_patch", False):
    _get_docs._helicopter_db_pipeline_patch = True  # type: ignore[attr-defined]
    LightevalTask.get_docs = _get_docs
    _evaluate._helicopter_db_pipeline_patch = True  # type: ignore[attr-defined]
    Pipeline.evaluate = _evaluate
    Pipeline._post_process_outputs = _post_process_outputs
    Pipeline.show_results = _show_results
    Pipeline.get_results = _get_results
    Pipeline.save_and_push_results = _save_and_push_results
