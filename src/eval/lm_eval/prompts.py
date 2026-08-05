from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence


GENERATION_PROMPTS = ("none", "open_think", "fake_think")
PROMPT_RENDERER_VERSION = 1


@dataclass(frozen=True)
class PromptProfile:
    name: str
    style: str | None
    stop: str | None

    @property
    def enabled(self) -> bool:
        return self.style is not None

    @property
    def sha256(self) -> str:
        contract = {
            "name": self.name,
            "style": self.style,
            "stop": self.stop,
            "renderer_version": PROMPT_RENDERER_VERSION,
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


PROMPT_PROFILES = {
    "none": PromptProfile("none", None, None),
    "bot": PromptProfile("bot", "bot", "✿"),
    "assistant": PromptProfile("assistant", "assistant", "\nUser:"),
    "function_calling": PromptProfile(
        "function_calling", "function_calling", "\n### User"
    ),
}


def get_prompt_profile(name: str) -> PromptProfile:
    try:
        return PROMPT_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            "prompt profile must be one of: " + ", ".join(PROMPT_PROFILES)
        ) from error


def render_prompt(
    profile: PromptProfile,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
    generation_prompt: str,
) -> str:
    if not profile.enabled:
        raise ValueError("the none prompt profile cannot render chat messages")
    if generation_prompt not in GENERATION_PROMPTS:
        raise ValueError(
            "generation prompt must be one of: " + ", ".join(GENERATION_PROMPTS)
        )

    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported RWKV prompt role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("RWKV prompt message content must be text")
        rendered.append(_render_message(profile, role, content))

    if add_generation_prompt:
        rendered.append(
            _render_generation(profile, _generation_text(generation_prompt))
        )
    separator = "\n\n" if profile.style == "assistant" else "\n"
    return separator.join(rendered)


def _render_message(profile: PromptProfile, role: str, content: str) -> str:
    if profile.style == "bot":
        label = "Bot" if role == "assistant" else role.title()
        return f"{label}✿{content}✿"
    if profile.style == "assistant":
        label = "Assistant" if role == "assistant" else role.title()
        return f"{label}: {content}" if content else f"{label}:"
    if profile.style == "function_calling":
        label = "Assistant" if role == "assistant" else role.title()
        return f"### {label}\n{content}" if content else f"### {label}"
    raise ValueError(f"unsupported RWKV prompt profile: {profile.name}")


def _render_generation(profile: PromptProfile, content: str) -> str:
    if profile.style == "bot":
        return f"Bot✿{content}"
    if profile.style == "assistant":
        return f"Assistant: {content}" if content else "Assistant:"
    if profile.style == "function_calling":
        return f"### Assistant\n{content}" if content else "### Assistant"
    raise ValueError(f"unsupported RWKV prompt profile: {profile.name}")


def _generation_text(mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "open_think":
        return "<think"
    if mode == "fake_think":
        return "<think></think"
    raise ValueError("unsupported RWKV generation prompt: " + mode)
