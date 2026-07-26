"""Architecture-neutral prompt protocol contracts.

The wrapper is selected by the evaluation run, not by an individual
benchmark TOML.  RWKV's current vLLM deployment consumes the naive protocol
through its native chat template; the normal protocol is sent as a request
template when the server explicitly trusts request-level templates.
"""

from __future__ import annotations


PROMPT_MODES = ("naive_cot", "naive_nocot", "normal_cot", "normal_nocot")

PROMPT_TEMPLATES = {
    "naive_cot": "User: {query}\n\nAssistant: <think",
    "naive_nocot": "User: {query}\n\nAssistant: <think></think",
    "normal_cot": "User\u273f{query}\u273f\nBot\u273f<think",
    "normal_nocot": "User\u273f{query}\u273f\nBot\u273f<think></think",
}

NORMAL_CHAT_TEMPLATES = {
    "normal_cot": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User\u273f{{ message['content'] }}\u273f"
        "{% elif message['role'] == 'assistant' %}Bot\u273f{{ message['content'] }}\u273f"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Bot\u273f<think{% endif %}"
    ),
    "normal_nocot": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User\u273f{{ message['content'] }}\u273f"
        "{% elif message['role'] == 'assistant' %}Bot\u273f{{ message['content'] }}\u273f"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Bot\u273f<think></think{% endif %}"
    ),
}


def prompt_template_for_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    try:
        return PROMPT_TEMPLATES[normalized]
    except KeyError as error:
        allowed = ", ".join(PROMPT_MODES)
        raise ValueError(f"prompt mode must be one of: {allowed}; got {mode!r}") from error


def normal_chat_template_for_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    try:
        return NORMAL_CHAT_TEMPLATES[normalized]
    except KeyError as error:
        raise ValueError(f"normal chat template requires a normal mode, got {mode!r}") from error


__all__ = [
    "NORMAL_CHAT_TEMPLATES",
    "PROMPT_MODES",
    "PROMPT_TEMPLATES",
    "normal_chat_template_for_mode",
    "prompt_template_for_mode",
]
