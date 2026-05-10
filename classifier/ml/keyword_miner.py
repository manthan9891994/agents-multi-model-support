"""Mine candidate L1 keywords from production decision logs.

Reads `routing_decisions.jsonl`, groups task previews by `task_type`, computes
the most distinctive 1- and 2-grams per group via a class-conditional log-odds
score, and returns the top suggestions that aren't already in any keyword pack.

Used by `dmr keywords suggest`.

Standalone, no sklearn — works with the lightweight install.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Words we never want to suggest as keywords — too generic to be useful
_STOPWORDS = frozenset(
    """a an and are as at be been being but by for from has have he her him his
    how i if in into is it its me my of on or our she that the their them then
    there these they this to was we were what when where which who why will with
    you your yours just like make made me my no not now only own same so some
    than that thats too very can could would should may might must shall do does
    did done could would should""".split()
)

_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_-]{2,}\b")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if n == 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _resolve_since(since: str | None) -> str | None:
    """Convert '7d', '24h', '4w' → ISO timestamp; pass through ISO strings."""
    if not since:
        return None
    s = since.strip().lower()
    units = {"h": "hours", "d": "days", "w": "weeks"}
    if s and s[-1] in units and s[:-1].isdigit():
        kw = {units[s[-1]]: int(s[:-1])}
        return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat()
    return since


def _existing_keywords() -> set[str]:
    """All keywords currently in any built-in or user-registered pack."""
    from classifier.layers.layer1.constants import _ESCALATORS, _TASK_KEYWORDS

    seen: set[str] = set()
    for groups in _TASK_KEYWORDS.values():
        for kws in (groups or {}).values():
            for kw in kws or []:
                seen.add(kw.lower())
    seen.update(_ESCALATORS.keys())
    return seen


def suggest_keywords(
    *,
    since: str | None = None,
    top_per_type: int = 15,
    min_occurrences: int = 3,
    smoothing: float = 1.0,
) -> dict[str, list[tuple[str, float, int]]]:
    """Return distinctive n-grams per task_type, sorted by class-conditional log-odds.

    Reads decisions via `decision_logger.read_decisions()` so any pluggable
    backend (Redis/Kafka/S3) is honored.

    Args:
        since:           Window like "7d" / "24h" / ISO timestamp. None = all data.
        top_per_type:    Max suggestions per task_type.
        min_occurrences: Drop n-grams seen fewer times in the source class.
        smoothing:       Add-k smoothing for the log-odds denominator (avoids div/0).

    Returns:
        {task_type: [(ngram, score, count), ...]} sorted by score descending.
        Score interpretation: log( P(ngram | this class) / P(ngram | other classes) ).
    """
    from classifier.infra.decision_logger import read_decisions

    since_iso = _resolve_since(since)
    rows = read_decisions(since=since_iso)

    # Group token-streams by task_type
    by_class: dict[str, list[list[str]]] = defaultdict(list)
    for r in rows:
        tt = r.get("task_type")
        text = r.get("task_preview") or r.get("task") or ""
        if not tt or not text:
            continue
        by_class[tt].append(_tokenize(text))

    if not by_class:
        return {}

    # Total class-document counts and global totals
    class_doc_count: dict[str, int] = {tt: len(docs) for tt, docs in by_class.items()}
    total_docs = sum(class_doc_count.values())

    # Per-class n-gram document frequency (unigrams + bigrams)
    class_ngram_df: dict[str, Counter] = {tt: Counter() for tt in by_class}
    for tt, docs in by_class.items():
        for tokens in docs:
            seen_in_doc: set[str] = set()
            for n in (1, 2):
                for ng in _ngrams(tokens, n):
                    seen_in_doc.add(ng)
            for ng in seen_in_doc:
                class_ngram_df[tt][ng] += 1

    # Global df = sum across all classes
    global_df: Counter = Counter()
    for df in class_ngram_df.values():
        global_df.update(df)

    existing = _existing_keywords()
    suggestions: dict[str, list[tuple[str, float, int]]] = {}

    for tt, df in class_ngram_df.items():
        scored: list[tuple[str, float, int]] = []
        n_in_class = class_doc_count[tt]
        n_other = total_docs - n_in_class
        if n_other <= 0:
            continue
        for ng, count in df.items():
            if count < min_occurrences:
                continue
            if ng in existing:
                continue
            # Log-odds: P(ng | class) vs. P(ng | not class), Laplace-smoothed
            p_in = (count + smoothing) / (n_in_class + smoothing * 2)
            df_other = global_df[ng] - count
            p_out = (df_other + smoothing) / (n_other + smoothing * 2)
            score = math.log(p_in / p_out)
            if score <= 0:
                continue  # not distinctive
            scored.append((ng, score, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        suggestions[tt] = scored[:top_per_type]

    return suggestions
