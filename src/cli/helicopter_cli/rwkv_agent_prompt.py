"""Naive-Chat prompt and context budgeting used by the local Agent adapter.

The reference RWKV FC runner renders an OpenAI conversation as a single
role-labelled transcript.  This module keeps that serialization local to the
evaluation adapter so the source EvalScope messages remain unchanged in the
trace and in the strict parser.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


TRUNCATION_NOTICE = "[Earlier conversation history truncated]"
DEFAULT_LONG_DOC_MIN_CHARS = 6000
DEFAULT_LONG_DOC_MAX_CHARS = 1000
DEFAULT_LONG_DOC_OVERLAP_LINES = 3
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS = 4
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS = 6000
RWKV_FLOWER_JSON_PROMPT_STYLE = "rwkv_flower_json"
_FLOWER = "\u273f"
_FLOWER_ASSISTANT_PREFIX = f"Bot{_FLOWER}<think></think>\n```json\n"


@dataclass(frozen=True, slots=True)
class LongContextConfig:
    min_long_text_chars: int = DEFAULT_LONG_DOC_MIN_CHARS
    max_chunk_chars: int = DEFAULT_LONG_DOC_MAX_CHARS
    overlap_lines: int = DEFAULT_LONG_DOC_OVERLAP_LINES
    max_evidence_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS
    max_evidence_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS


def normalize_rwkv_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    return re.sub(r"\n{2,}", "\n", normalized)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _message_content(message: Mapping[str, object]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is not None:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False, separators=(",", ":"))
    return ""


def normalize_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = _message_content(message)
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _query_terms(text: str) -> tuple[str, ...]:
    latin = re.findall(r"[a-z0-9_]{2,}", text.lower())
    cjk = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}", text)
    return tuple(dict.fromkeys([*latin, *cjk]))


def _infer_query(
    messages: Sequence[Mapping[str, str]],
    *,
    max_chars: int = 1200,
    skip_longer_than: int = DEFAULT_LONG_DOC_MIN_CHARS,
) -> str:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = normalize_rwkv_text(message["content"])
        if content and len(content) < max(1, int(skip_longer_than)):
            return content[-max_chars:]
    for message in reversed(messages):
        content = normalize_rwkv_text(message["content"])
        if content:
            return content[-max_chars:]
    return ""


def _chunks(text: str, *, max_chars: int, overlap_lines: int) -> list[tuple[int, int, int, str]]:
    lines = normalize_rwkv_text(text).splitlines() or [""]
    base: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_chars = 0
    start_line = 1
    for line_no, line in enumerate(lines, start=1):
        pieces = [line[index : index + max_chars] for index in range(0, len(line), max_chars)] or [""]
        for piece in pieces:
            if current and current_chars + len(piece) + 1 > max_chars:
                base.append((start_line, line_no - 1, "\n".join(current)))
                current = []
                current_chars = 0
                start_line = line_no
            current.append(piece)
            current_chars += len(piece) + (1 if len(current) > 1 else 0)
    if current:
        base.append((start_line, len(lines), "\n".join(current)))

    output: list[tuple[int, int, int, str]] = []
    for index, (line_start, line_end, body) in enumerate(base):
        prefix_lines: list[str] = []
        overlap = 0
        if index and overlap_lines:
            previous = base[index - 1][2].splitlines()
            prefix_lines = previous[-overlap_lines:]
            overlap = len(prefix_lines)
            line_start = max(base[index - 1][0], base[index - 1][1] - overlap + 1)
        emitted = "\n".join([*prefix_lines, body]) if prefix_lines else body
        output.append((index, line_start, line_end, emitted))
    return output


def _compact_long_message(content: str, *, query: str, config: LongContextConfig) -> tuple[str, dict[str, Any] | None]:
    normalized = normalize_rwkv_text(content)
    if len(normalized) < max(1, int(config.min_long_text_chars)):
        return normalized, None
    chunks = _chunks(
        normalized,
        max_chars=max(1, int(config.max_chunk_chars)),
        overlap_lines=max(0, int(config.overlap_lines)),
    )
    terms = _query_terms(query)
    scored: list[tuple[float, int, int, int, str]] = []
    for chunk_id, line_start, line_end, chunk_text in chunks:
        lowered = chunk_text.lower()
        score = float(sum(lowered.count(term.lower()) for term in terms))
        if score > 0 or not terms:
            scored.append((score, chunk_id, line_start, line_end, chunk_text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[float, int, int, int, str]] = []
    used_chars = 0
    for item in scored:
        if len(selected) >= max(1, int(config.max_evidence_chunks)):
            break
        if selected and used_chars + len(item[4]) > max(1, int(config.max_evidence_chars)):
            continue
        selected.append(item)
        used_chars += len(item[4])
        if used_chars >= max(1, int(config.max_evidence_chars)):
            break
    selected.sort(key=lambda item: item[1])
    header = (
        f"[Long document compacted: original_chars={len(normalized)}; chunks={len(chunks)}; "
        f"selected_chunks={len(selected)}; mode=lexical; reason=lexical]"
    )
    if not selected:
        compacted = header + "\n[No evidence chunk selected.]"
    else:
        parts = [header]
        for score, chunk_id, line_start, line_end, chunk_text in selected:
            parts.append(f"[chunk {chunk_id} lines {line_start}-{line_end} score={score:.3f}]\n{chunk_text.strip()}")
        compacted = "\n\n".join(parts)
    return compacted, {
        "original_chars": len(normalized),
        "chunk_count": len(chunks),
        "selected_chunk_ids": [item[1] for item in selected],
        "compacted": True,
    }


def compact_messages_for_long_context(
    messages: Sequence[Mapping[str, object]],
    *,
    config: LongContextConfig | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    cfg = config or LongContextConfig()
    normalized = normalize_messages(messages)
    query = _infer_query(normalized, skip_longer_than=cfg.min_long_text_chars)
    compacted: list[dict[str, str]] = []
    selected: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(normalized):
        if message["role"] == "system":
            # The evaluation adapter may add a routing system block, but it
            # must never rewrite the source system prompt or its semantics.
            compacted.append(dict(message))
            continue
        content, trace = _compact_long_message(message["content"], query=query, config=cfg)
        compacted.append({"role": message["role"], "content": content})
        if trace is not None:
            selected[str(index)] = trace
    return compacted, {
        "query_chars": len(query),
        "compacted_message_count": len(selected),
        "selected_messages": selected,
        "config": asdict(cfg),
    }


def _rendered_message_len(message: Mapping[str, str]) -> int:
    return len(message["role"]) + len(message["content"]) + 4


def _fit_message_tail(message: Mapping[str, str], budget: int) -> dict[str, str] | None:
    role = str(message.get("role") or "user")
    content = str(message.get("content") or "")
    overhead = len(role) + 4
    if budget <= overhead:
        return None
    content_budget = budget - overhead
    return {"role": role, "content": content[-content_budget:]}


def trim_message_history(
    messages: Sequence[Mapping[str, object]],
    *,
    max_chars: int,
) -> tuple[list[dict[str, str]], bool]:
    normalized = normalize_messages(messages)
    if max_chars <= 0 or not normalized:
        return [], bool(normalized)
    source_system = [message for message in normalized if message["role"] == "system"]
    source_history = [message for message in normalized if message["role"] != "system"]
    # System messages are source instructions, not disposable history. Keep
    # their exact content even when the remaining history has to be reduced.
    system_chars = sum(_rendered_message_len(message) for message in source_system)
    history_budget = max(0, int(max_chars) - system_chars)
    if source_system and history_budget <= 0:
        return source_system, bool(source_history)
    if source_system:
        bounded_history, truncated = _trim_non_system_history(source_history, max_chars=history_budget)
        selected_by_index: dict[int, dict[str, str]] = {}
        consumed: set[int] = set()
        synthetic: list[dict[str, str]] = []
        for item in bounded_history:
            match_index = None
            for index, source in enumerate(source_history):
                if index in consumed or source["role"] != item["role"]:
                    continue
                if source["content"] == item["content"] or (
                    len(item["content"]) < len(source["content"])
                    and source["content"].endswith(item["content"])
                ):
                    match_index = index
                    break
            if match_index is None:
                synthetic.append(item)
            else:
                consumed.add(match_index)
                selected_by_index[match_index] = item
        first_selected = min(consumed) if consumed else None
        result: list[dict[str, str]] = []
        history_index = 0
        for message in normalized:
            if message["role"] == "system":
                result.append(dict(message))
                continue
            if first_selected == history_index and synthetic:
                result.extend(synthetic)
            replacement = selected_by_index.get(history_index)
            if replacement is not None:
                result.append(replacement)
            history_index += 1
        if synthetic and first_selected is None:
            result.extend(synthetic)
        return result, truncated
    return _trim_non_system_history(normalized, max_chars=max_chars)


def _trim_non_system_history(
    normalized: Sequence[Mapping[str, str]],
    *,
    max_chars: int,
) -> tuple[list[dict[str, str]], bool]:
    if max_chars <= 0 or not normalized:
        return [], bool(normalized)
    total = 0
    kept_reversed: list[dict[str, str]] = []
    cut_index = -1
    for index in range(len(normalized) - 1, -1, -1):
        message = normalized[index]
        rendered_len = _rendered_message_len(message)
        if total + rendered_len > max_chars:
            cut_index = index
            break
        kept_reversed.append(message)
        total += rendered_len
    kept = list(reversed(kept_reversed))
    if cut_index < 0:
        return kept, False

    available = max_chars - total
    notice_message = {"role": "user", "content": TRUNCATION_NOTICE}
    notice_len = _rendered_message_len(notice_message)
    prefix: list[dict[str, str]] = []
    truncated = None
    if available > notice_len:
        truncated = _fit_message_tail(normalized[cut_index], available - notice_len)
        if truncated is not None:
            prefix.append(notice_message)
    if truncated is None:
        truncated = _fit_message_tail(normalized[cut_index], available)
        if truncated is None and kept and total + notice_len <= max_chars:
            prefix.append(notice_message)
    if truncated is not None:
        prefix.append(truncated)
    return [*prefix, *kept], True


def _strip_role_prefix(content: str, prefix: str) -> str:
    normalized = normalize_rwkv_text(content)
    return normalized[len(prefix) :].lstrip() if normalized.startswith(prefix) else normalized


def _looks_like_json(content: str) -> bool:
    stripped = normalize_rwkv_text(content)
    return stripped.startswith(("{", "[", "```json", "```JSON", "Assistant: ```json"))


def _render_user(content: str) -> str:
    return f"User: {_strip_role_prefix(content, 'User:')}".rstrip()


def _render_assistant(content: str, *, assistant_prefix: str) -> str:
    normalized = _strip_role_prefix(content, "Assistant:")
    if _looks_like_json(normalized):
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            normalized = "\n".join(lines).strip()
        return f"{assistant_prefix}{normalized}\n```"
    return f"Assistant: {normalized}".rstrip()


def _render_flower_user(content: str) -> str:
    return f"User{_FLOWER}{_strip_role_prefix(content, 'User:')}{_FLOWER}".rstrip()


def _render_flower_system(content: str) -> str:
    return f"User{_FLOWER}System:\n{normalize_rwkv_text(content)}{_FLOWER}".rstrip()


def _render_flower_assistant(content: str, *, assistant_prefix: str) -> str:
    normalized = _strip_role_prefix(content, "Assistant:")
    if _looks_like_json(normalized):
        if normalized.startswith("```"):
            lines = normalized.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            normalized = "\n".join(lines).strip()
        return f"{assistant_prefix}{normalized}\n```{_FLOWER}"
    return f"Bot{_FLOWER}{normalized}{_FLOWER}".rstrip()


def build_rwkv_json_call_prompt(
    system_prompt: str,
    messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
    prompt_max_chars: int,
    assistant_prefix: str | None = None,
    prompt_style: str | None = None,
) -> tuple[str, dict[str, Any]]:
    style = str(prompt_style or "assistant").strip().lower()
    if style not in {"assistant", RWKV_FLOWER_JSON_PROMPT_STYLE}:
        raise ValueError(f"unsupported RWKV prompt style: {prompt_style!r}")
    if assistant_prefix is None:
        assistant_prefix = (
            _FLOWER_ASSISTANT_PREFIX
            if style == RWKV_FLOWER_JSON_PROMPT_STYLE
            else "Assistant: <think></think>\n```json\n"
        )
    bounded, history_truncated = trim_message_history(messages, max_chars=max(0, int(history_max_chars)))
    parts = [
        _render_flower_system(system_prompt)
        if style == RWKV_FLOWER_JSON_PROMPT_STYLE
        else f"System: {normalize_rwkv_text(system_prompt)}".rstrip()
    ]
    for message in bounded:
        content = normalize_rwkv_text(message["content"])
        if not content:
            continue
        role = message["role"]
        if role in {"tool", "function", "observation"}:
            parts.append(
                _render_flower_user("Function output:\n" + content)
                if style == RWKV_FLOWER_JSON_PROMPT_STYLE
                else _render_user("Function output:\n" + content)
            )
        elif role == "assistant":
            parts.append(
                _render_flower_assistant(content, assistant_prefix=assistant_prefix)
                if style == RWKV_FLOWER_JSON_PROMPT_STYLE
                else _render_assistant(content, assistant_prefix=assistant_prefix)
            )
        elif role == "system":
            # Keep source system content verbatim and in a System-labelled block.
            parts.append(
                _render_flower_system(content)
                if style == RWKV_FLOWER_JSON_PROMPT_STYLE
                else f"System: {content}".rstrip()
            )
        else:
            parts.append(
                _render_flower_user(content)
                if style == RWKV_FLOWER_JSON_PROMPT_STYLE
                else _render_user(content)
            )
    parts.append(assistant_prefix)
    prompt = "\n\n".join(parts)
    if len(prompt) > max(1, int(prompt_max_chars)) and history_max_chars > 1:
        overflow = len(prompt) - int(prompt_max_chars)
        reduced_history = max(1, int(history_max_chars) - overflow - 512)
        if reduced_history < int(history_max_chars):
            return build_rwkv_json_call_prompt(
                system_prompt,
                messages,
                history_max_chars=reduced_history,
                prompt_max_chars=prompt_max_chars,
                assistant_prefix=assistant_prefix,
                prompt_style=style,
            )
    return prompt, {
        "history_max_chars": int(history_max_chars),
        "history_truncated": history_truncated,
        "prompt_chars": len(prompt),
        "prompt_over_budget": len(prompt) > max(1, int(prompt_max_chars)),
    }


__all__ = [
    "DEFAULT_LONG_DOC_MAX_CHARS",
    "DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS",
    "RWKV_FLOWER_JSON_PROMPT_STYLE",
    "DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS",
    "DEFAULT_LONG_DOC_MIN_CHARS",
    "DEFAULT_LONG_DOC_OVERLAP_LINES",
    "LongContextConfig",
    "build_rwkv_json_call_prompt",
    "compact_messages_for_long_context",
    "normalize_messages",
    "normalize_rwkv_text",
    "trim_message_history",
]
