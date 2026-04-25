import logging
import re
import time
from typing import Optional

from classifier.core.exceptions import ClassificationError
from classifier.core.types import TaskType, TaskComplexity, ModelTier
from classifier.core.registry import TIER_MATRIX

logger = logging.getLogger(__name__)


# ── Keyword tables ─────────────────────────────────────────────────────────────
# Lists sorted by length DESC within each group — greedy matching ensures longer
# phrases ("write a function") match before shorter ones ("write").
# primary=3pts, secondary=1pt.

_TASK_KEYWORDS: dict[TaskType, dict[str, list[str]]] = {
    TaskType.CONVERSATION: {
        "primary": ["good morning", "good evening", "how are you", "what's up",
                    "thank you", "thanks", "hello", "hey", "bye", "goodbye"],
        "secondary": ["sounds good", "got it", "okay", "sure", "ok"],
    },
    TaskType.MATH: {
        "primary": ["calculate", "compute", "solve", "integral", "derivative",
                    "equation", "matrix", "factorial", "probability", "statistics",
                    "calculus", "algebra", "arithmetic"],
        "secondary": ["sum", "average", "mean", "variance", "formula", "proof"],
    },
    TaskType.TRANSLATION: {
        "primary": ["convert to language", "in spanish", "in french", "in german",
                    "in japanese", "in chinese", "in portuguese", "in hindi",
                    "translate", "localize", "multilingual"],
        "secondary": ["language", "locale", "i18n", "l10n"],
    },
    TaskType.MULTIMODAL: {
        "primary": ["analyze this image", "describe the photo", "analyze this photo",
                    "what's in this image", "image recognition", "audio transcription",
                    "speech to text", "extract text from", "vision task",
                    "transcribe this", "ocr this"],
        "secondary": ["image", "photo", "picture", "audio", "video",
                      "transcribe", "ocr", "vision"],
    },
    TaskType.REASONING: {
        "primary": ["pros and cons", "difference between", "which is better",
                    "analyze the", "trade-off", "tradeoff", "compare", "evaluate",
                    "assess", "debate", "interpret", "distinguish", "contrast",
                    "critically"],
        "secondary": ["why", "should i", "versus", "vs", "argument", "logic",
                      "validate", "review"],
    },
    TaskType.THINKING: {
        "primary": ["system design", "approach to", "what would be", "how should i",
                    "plan", "strategy", "brainstorm", "design", "architect",
                    "roadmap", "workflow"],
        "secondary": ["ideate", "organize", "structure", "process", "explore"],
    },
    TaskType.ANALYZING: {
        "primary": ["insights from", "analyze data", "find pattern", "trend analysis",
                    "correlation", "anomaly", "distribution", "benchmark",
                    "seasonal", "statistical"],
        "secondary": ["data", "metric", "aggregate", "breakdown", "chart", "graph",
                      "dataset", "measure", "trend"],
    },
    TaskType.CODE_CREATION: {
        "primary": ["write a script", "write a test", "write a class",
                    "write a function", "write code", "api endpoint", "unit test",
                    "fix bug", "implement", "debug", "refactor"],
        "secondary": ["function", "algorithm", "optimize", "class", "code",
                      "develop", "programming", "build"],
    },
    TaskType.DOC_CREATION: {
        "primary": ["create documentation", "write a summary", "write a report",
                    "write a tutorial", "write a guide", "write a readme",
                    "write documentation", "write the"],
        "secondary": ["document", "readme", "guide", "manual", "summarize",
                      "describe", "explain", "comment"],
    },
}

# Suppress a category score (−2 per hit)
_NEGATIVE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_CREATION: ["explain", "what is", "tell me about", "how does",
                              "describe what"],
    TaskType.REASONING:     ["calculate", "compute", "solve for"],
    TaskType.MATH:          ["write", "implement", "create", "design"],
}

# Negation prefixes — presence before a keyword suppresses its score
_NEGATION_RE = re.compile(
    r"\b(don't|dont|do not|not|without|no|instead of|rather than|avoid)\b",
    re.IGNORECASE,
)

# De-escalators — push complexity DOWN one level
_DEESCALATORS = {
    "simple", "basic", "quick", "brief", "one-line", "trivial",
    "short", "easy", "beginner", "just a", "small", "example only",
    "in one sentence", "in a few words", "one-liner", "tldr", "just the gist",
}

# Weighted escalators — sum weights:
#   >= 1 → +1 level   >= 3 → +2 levels   >= 5 → RESEARCH
_ESCALATORS: dict[str, int] = {
    "distributed":         3,
    "microservices":       3,
    "production-ready":    3,
    "high availability":   3,
    "fault-tolerant":      3,
    "across 10":           3,
    "multiple industries": 3,
    "enterprise":          2,
    "thread-safe":         2,
    "concurrent":          2,
    "lru":                 2,
    "eviction":            2,
    "oauth":               2,
    "in-depth":            2,
    "end-to-end":          2,
    "across multiple":     2,
    "market data":         2,
    "scalable":            2,
    "architecture":        2,
    "ttl":                 1,
    "authentication":      1,
    "authorization":       1,
    "rest api":            1,
    "comprehensive":       1,
    "detailed":            1,
    "thorough":            1,
    "advanced":            1,
}

# Algorithm / data-structure names → minimum COMPLEX complexity
_ALGORITHM_NAMES = {
    "raft", "paxos", "b-tree", "bloom filter", "consistent hashing",
    "red-black tree", "avl tree", "dijkstra", "byzantine fault",
    "two-phase commit", "saga pattern", "cqrs", "event sourcing",
    "merkle tree", "lsm tree", "skip list", "chord protocol",
    "vector clock", "crdt",
}

# Domain escalation — force a minimum tier based on domain sensitivity
_DOMAIN_MIN_TIER: dict[str, ModelTier] = {
    "clinical":              ModelTier.MEDIUM,
    "diagnosis":             ModelTier.MEDIUM,
    "ehr":                   ModelTier.MEDIUM,
    "hipaa":                 ModelTier.HIGH,
    "patient data":          ModelTier.HIGH,
    "treatment":             ModelTier.MEDIUM,
    "contract":              ModelTier.MEDIUM,
    "liability":             ModelTier.MEDIUM,
    "compliance":            ModelTier.MEDIUM,
    "gdpr":                  ModelTier.HIGH,
    "legal":                 ModelTier.MEDIUM,
    "litigation":            ModelTier.HIGH,
    "portfolio":             ModelTier.MEDIUM,
    "risk model":            ModelTier.HIGH,
    "hedge fund":            ModelTier.HIGH,
    "derivative pricing":    ModelTier.HIGH,
    "financial regulation":  ModelTier.HIGH,
}

# Format-only requests — suppress complexity escalation
_FORMAT_REQUESTS = {
    "return json", "as json", "in json format", "json output",
    "as a table", "in table format", "formatted as",
    "in bullet points", "as a list", "in yaml", "as csv",
    "as markdown", "in markdown", "as xml",
}

# Question patterns → force SIMPLE regardless of escalators
_YES_NO_RE = re.compile(
    r"^(can|could|should|would|is|are|does|do|has|have|will|was|were)\s",
    re.IGNORECASE,
)
_WHAT_IS_RE = re.compile(r"^what (is|are|was|were)\s", re.IGNORECASE)

# Context window for LOW tier — inputs > 50% of this get bumped SIMPLE→STANDARD
_LOW_TIER_CONTEXT_TOKENS = 8_192

_TIER_ORDER = [ModelTier.LOW, ModelTier.MEDIUM, ModelTier.HIGH]
_COMPLEXITY_LEVELS = [
    TaskComplexity.SIMPLE, TaskComplexity.STANDARD,
    TaskComplexity.COMPLEX, TaskComplexity.RESEARCH,
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split()) * 4 // 3


def _detect_language(text: str) -> str:
    if not text:
        return "en"
    total = len(text)
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")

    if cjk / total > 0.15:
        return "zh"
    if arabic / total > 0.15:
        return "ar"
    if cyrillic / total > 0.15:
        return "ru"
    if devanagari / total > 0.15:
        return "hi"
    return "en"


def _negation_positions(lower: str) -> set[int]:
    positions: set[int] = set()
    for m in _NEGATION_RE.finditer(lower):
        end = m.end()
        positions.update(range(end, min(end + 50, len(lower))))
    return positions


# ── Core classifiers ───────────────────────────────────────────────────────────

def _detect_task_type(lower: str) -> tuple[TaskType, float, bool]:
    scores: dict[TaskType, float] = {t: 0.0 for t in TaskType}
    negated = _negation_positions(lower)

    for task_type, groups in _TASK_KEYWORDS.items():
        for weight, group_key in [(3.0, "primary"), (1.0, "secondary")]:
            sorted_kws = sorted(groups.get(group_key, []), key=len, reverse=True)
            consumed: set[int] = set()
            for kw in sorted_kws:
                for m in re.finditer(r"\b" + re.escape(kw) + r"\b", lower):
                    s, e = m.start(), m.end()
                    region = set(range(s, e))
                    if region & consumed:
                        continue
                    mid = (s + e) // 2
                    if mid in negated:
                        scores[task_type] -= weight * 0.5
                    else:
                        scores[task_type] += weight
                        consumed |= region

    for task_type, neg_kws in _NEGATIVE_KEYWORDS.items():
        for kw in sorted(neg_kws, key=len, reverse=True):
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                scores[task_type] -= 2.0

    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return TaskType.DOC_CREATION, 0.3, False

    positives = sorted(
        [(t, s) for t, s in scores.items() if s > 0],
        key=lambda x: x[1], reverse=True,
    )
    is_ambiguous = (
        len(positives) >= 2
        and positives[1][1] / positives[0][1] >= 0.80
    )

    total = sum(v for v in scores.values() if v > 0) or 1
    confidence = min(scores[best] / total + 0.2, 1.0)
    if is_ambiguous:
        confidence = min(confidence, 0.45)

    return best, confidence, is_ambiguous


def _detect_complexity(lower: str, tokens: int) -> TaskComplexity:
    is_format_only = any(fr in lower for fr in _FORMAT_REQUESTS)
    has_algorithm  = any(alg in lower for alg in _ALGORITHM_NAMES)

    if tokens > 15_000:
        base = TaskComplexity.RESEARCH
    elif tokens > 5_000:
        base = TaskComplexity.COMPLEX
    elif tokens > 500:
        base = TaskComplexity.STANDARD
    else:
        base = TaskComplexity.SIMPLE

    if base == TaskComplexity.SIMPLE and tokens > _LOW_TIER_CONTEXT_TOKENS * 0.5:
        base = TaskComplexity.STANDARD

    idx = _COMPLEXITY_LEVELS.index(base)

    if has_algorithm:
        idx = max(idx, 2)

    if not is_format_only:
        weight = sum(w for kw, w in _ESCALATORS.items() if kw in lower)
        if weight >= 5:
            idx = min(idx + 3, 3)
        elif weight >= 3:
            idx = min(idx + 2, 3)
        elif weight >= 1:
            idx = min(idx + 1, 3)

    complexity = _COMPLEXITY_LEVELS[idx]

    if sum(1 for kw in _DEESCALATORS if kw in lower) >= 1 and idx > 0:
        complexity = _COMPLEXITY_LEVELS[idx - 1]

    stripped = lower.strip()
    if _YES_NO_RE.match(stripped) or _WHAT_IS_RE.match(stripped):
        if complexity in (TaskComplexity.STANDARD, TaskComplexity.COMPLEX):
            complexity = TaskComplexity.SIMPLE

    return complexity


def _domain_min_tier(lower: str) -> Optional[ModelTier]:
    result: Optional[ModelTier] = None
    for kw, min_t in _DOMAIN_MIN_TIER.items():
        if kw in lower:
            if result is None or _TIER_ORDER.index(min_t) > _TIER_ORDER.index(result):
                result = min_t
    return result


# ── Public entry point ─────────────────────────────────────────────────────────

def classify_layer1(
    task: str,
    history: Optional[list[str]] = None,
) -> tuple[TaskType, TaskComplexity, ModelTier, float, str]:
    """Layer 1 — keyword + heuristic classifier (<1ms, no external calls).

    Args:
        task:    User's input text.
        history: Optional prior conversation turns (most-recent last).

    Returns:
        (task_type, complexity, tier, confidence, reasoning)
    """
    if not task or not task.strip():
        raise ClassificationError("Layer 1 received an empty task string.")

    t0    = time.perf_counter()
    lower = task.lower()
    tokens = _count_tokens(task)

    task_type, confidence, is_ambiguous = _detect_task_type(lower)

    # History bias: recent code-heavy turns nudge toward CODE_CREATION
    if history and task_type != TaskType.CODE_CREATION:
        recent = " ".join(h.lower() for h in history[-3:])
        code_signals = sum(
            1 for kw in ("implement", "function", "debug", "code", "write a")
            if kw in recent
        )
        if code_signals >= 2:
            task_type  = TaskType.CODE_CREATION
            confidence = min(confidence + 0.1, 1.0)

    lang = _detect_language(task)
    if lang != "en":
        confidence   = min(confidence, 0.40)
        is_ambiguous = True

    complexity    = _detect_complexity(lower, tokens)
    tier          = TIER_MATRIX.get((task_type, complexity), ModelTier.MEDIUM)
    domain_floor  = _domain_min_tier(lower)

    if domain_floor and _TIER_ORDER.index(domain_floor) > _TIER_ORDER.index(tier):
        tier = domain_floor

    elapsed = (time.perf_counter() - t0) * 1000

    flags: list[str] = []
    if is_ambiguous:
        flags.append("ambiguous")
    if lang != "en":
        flags.append(f"lang={lang}")
    if domain_floor:
        flags.append(f"domain_min={domain_floor.value}")

    reason = (
        f"layer1 | type={task_type.value} complexity={complexity.value} "
        f"tokens={tokens} tier={tier.value} ({elapsed:.1f}ms)"
        + (f" [{', '.join(flags)}]" if flags else "")
    )
    logger.debug(reason)
    return task_type, complexity, tier, confidence, reason
