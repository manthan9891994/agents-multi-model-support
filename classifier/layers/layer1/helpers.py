import re

_CHARS_PER_TOKEN: dict[str, float] = {
    "google":    4.0,
    "anthropic": 4.5,
    "openai":    4.0,
}

_TRIVIAL_ACKS = {"k", "ok", "okay", "thx", "ty", "ack", "noted", "sure", "yep", "nope", "yes", "no"}

_NEGATION_RE = re.compile(
    r"\b(don't|dont|do not|not|without|no|instead of|rather than|avoid)\b",
    re.IGNORECASE,
)

_YES_NO_RE  = re.compile(r"^(can|could|should|would|is|are|does|do|has|have|will|was|were)\s", re.IGNORECASE)
_WHAT_IS_RE = re.compile(r"^what (is|are|was|were)\s", re.IGNORECASE)

_CONTINUATION_RE = re.compile(
    r"^(now|also|instead|but|and|then|next|"
    r"make it|try (it )?(again|once more)|"
    r"do (the )?same|like that|in that case|"
    r"what about|how about|can you also)\b",
    re.IGNORECASE,
)


def _is_trivial(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= 2:
        return True
    if not any(c.isalnum() for c in stripped):
        return True
    if stripped.lower() in _TRIVIAL_ACKS:
        return True
    return False


def _detect_language(text: str) -> str:
    if not text:
        return "en"
    total = len(text)
    cjk        = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
    arabic     = sum(1 for c in text if "؀" <= c <= "ۿ")
    cyrillic   = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    if cjk / total > 0.15:        return "zh"
    if arabic / total > 0.15:     return "ar"
    if cyrillic / total > 0.15:   return "ru"
    if devanagari / total > 0.15: return "hi"
    return "en"


def _negation_positions(lower: str) -> set[int]:
    positions: set[int] = set()
    for m in _NEGATION_RE.finditer(lower):
        end = m.end()
        positions.update(range(end, min(end + 50, len(lower))))
    return positions


def _count_tokens(text: str, provider: str = "google") -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        raw = len(enc.encode(text))
        if provider == "anthropic":
            return int(raw * 1.05)
        return raw
    except ImportError:
        cpt = _CHARS_PER_TOKEN.get(provider, 4.0)
        return int(len(text) / cpt)


def _extract_instruction(text: str) -> str:
    last_q = text.rfind("?")
    if last_q != -1:
        start = text.rfind("\n", 0, last_q)
        start = start + 1 if start != -1 else 0
        end   = text.find("\n", last_q)
        end   = end if end != -1 else len(text)
        instruction = text[start:end].strip()
        if len(instruction) > 10:
            return instruction

    code_start = text.find("```")
    if code_start != -1 and code_start > 20:
        return text[:code_start].strip()

    if len(text) > 2000:
        last_newline = text.rfind("\n")
        if last_newline > len(text) - 200:
            tail = text[last_newline:].strip()
            if len(tail) > 10:
                return tail
        return text[:200].strip()

    return text
