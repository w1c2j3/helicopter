"""Experimental EvalScope Tau2 compatibility shim.

EvalScope 1.9.1's Tau2 adapter assumes ModelOutput.usage is always present.
RWKV tool-call responses can legitimately omit that optional performance
metadata.  This shim is injected only into an experiment process and fills
the metadata with zero counters; it does not alter messages, tool calls,
arguments, model text, extraction, or judging.
"""

from __future__ import annotations

from functools import wraps


def _install() -> None:
    try:
        from evalscope.api.model.model_output import ModelUsage
        from evalscope.models.openai_compatible import OpenAICompatibleAPI
    except Exception:
        return

    original = OpenAICompatibleAPI.generate
    if getattr(original, "_helicopter_tau2_usage_compat", False):
        return

    @wraps(original)
    def generate(*args, **kwargs):
        output = original(*args, **kwargs)
        if output is not None and output.usage is None:
            output.usage = ModelUsage()
        return output

    generate._helicopter_tau2_usage_compat = True
    OpenAICompatibleAPI.generate = generate


_install()
