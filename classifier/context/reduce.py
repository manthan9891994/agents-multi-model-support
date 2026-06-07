"""Context reduction — the biggest agentic cost lever (input = ~72% of the bill).

Most of an agent's cost is the *accumulating context resent on every step*. This
prunes the neutral `contents` list before the call: keep the system prompt + the
originating user question + the last N tool results (truncated), and drop older
tool results. Pure function over a framework-neutral list so any adapter can use
it. Off by default (`DMR_CONTEXT_REDUCTION=prune`).

`contents` item schema: ``{"role": "system"|"user"|"model"|"tool", "text": str}``
(adapters convert framework messages to this shape and back).
"""

from __future__ import annotations

_CONTEXT_PREFIX = "for context"


def prune_context(contents, keep_last_tool_results: int = 3, max_tool_chars: int = 1500):
    """Return a trimmed copy of `contents`. Quality-neutral when tools' recent
    output is what matters; drops stale tool results that inflate input cost."""
    if not contents:
        return contents

    tool_positions = [i for i, c in enumerate(contents) if (c or {}).get("role") == "tool"]
    keep_tools = set(tool_positions[-keep_last_tool_results:]) if keep_last_tool_results > 0 else set()

    out: list = []
    seen_question = False
    for i, c in enumerate(contents):
        role = (c or {}).get("role")
        text = (c or {}).get("text", "") or ""
        if role == "system":
            out.append(c)
        elif role == "user":
            low = text.strip().lower()
            if low.startswith(_CONTEXT_PREFIX):
                continue  # drop framework wrapper noise
            if not seen_question:
                seen_question = True
            out.append(c)
        elif role == "tool":
            if i in keep_tools:
                if len(text) > max_tool_chars:
                    c = {**c, "text": text[:max_tool_chars] + " …[truncated]"}
                out.append(c)
            # else: drop stale tool result
        else:  # model / other — keep
            out.append(c)
    return out
