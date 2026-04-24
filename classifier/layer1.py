import logging
import re
import time

from classifier.exceptions import ClassificationError
from classifier.types import TaskType, TaskComplexity, ModelTier
from classifier.registry import TIER_MATRIX

logger = logging.getLogger(__name__)


# ── Keyword tables ────────────────────────────────────────────────────────────
# primary keywords score 3, secondary score 1.
# Multi-word phrases matched before single words to avoid partial overlaps.

_TASK_KEYWORDS: dict[TaskType, dict[str, list[str]]] = {
    TaskType.CONVERSATION: {
        "primary": ["hello", "hi ", "hey ", "how are you", "thanks", "thank you",
                    "good morning", "good evening", "bye", "goodbye", "what's up"],
        "secondary": ["okay", "ok", "sure", "got it", "sounds good"],
    },
    TaskType.MATH: {
        "primary": ["calculate", "compute", "solve", "integral", "derivative",
                    "equation", "matrix", "factorial", "probability", "statistics",
                    "calculus", "algebra", "arithmetic"],
        "secondary": ["sum", "average", "mean", "variance", "formula", "proof"],
    },
    TaskType.TRANSLATION: {
        "primary": ["translate", "in spanish", "in french", "in german", "in japanese",
                    "in chinese", "in portuguese", "localize", "convert to language",
                    "multilingual", "in hindi"],
        "secondary": ["language", "locale", "i18n", "l10n"],
    },
    TaskType.REASONING: {
        "primary": ["compare", "evaluate", "assess", "debate", "interpret",
                    "distinguish", "contrast", "pros and cons", "trade-off",
                    "tradeoff", "difference between", "which is better",
                    "analyze the", "critically"],
        "secondary": ["why", "should i", "versus", "vs", "argument", "logic",
                      "validate", "review"],
    },
    TaskType.THINKING: {
        "primary": ["plan", "strategy", "brainstorm", "design", "architect",
                    "system design", "roadmap", "workflow", "how should i",
                    "what would be", "approach to"],
        "secondary": ["ideate", "organize", "structure", "process", "explore"],
    },
    TaskType.ANALYZING: {
        "primary": ["analyze data", "find pattern", "trend analysis", "correlation",
                    "anomaly", "distribution", "benchmark", "seasonal", "insights from",
                    "statistical"],
        "secondary": ["data", "metric", "aggregate", "breakdown", "chart", "graph",
                      "dataset", "measure", "trend"],
    },
    TaskType.CODE_CREATION: {
        "primary": ["implement", "debug", "refactor", "write code", "fix bug",
                    "write a function", "write a class", "unit test", "write a test",
                    "api endpoint", "write a script"],
        "secondary": ["function", "algorithm", "optimize", "class", "code", "develop",
                      "programming", "build"],
    },
    TaskType.DOC_CREATION: {
        "primary": ["write a readme", "write documentation", "write a guide",
                    "write a tutorial", "write a report", "write a summary",
                    "write the", "create documentation"],
        "secondary": ["document", "readme", "guide", "manual", "summarize",
                      "describe", "explain", "comment"],
    },
}

# Keywords that suppress a category (negative scoring: -2 per hit)
_NEGATIVE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_CREATION: ["explain", "what is", "tell me about", "how does",
                              "describe what"],
    TaskType.REASONING:     ["calculate", "compute", "what is 2", "solve for"],
    TaskType.MATH:          ["write", "implement", "create", "design"],
}

# De-escalators — push complexity DOWN by one level
_DEESCALATORS = {
    "simple", "basic", "quick", "brief", "one-line", "trivial",
    "short", "easy", "beginner", "just a", "small", "example only",
    "in one sentence", "in a few words",
}

# Weighted escalators — sum weights to determine complexity jump
# weight >= 3 → COMPLEX; weight >= 5 → RESEARCH
_ESCALATORS: dict[str, int] = {
    "distributed":      3,
    "microservices":    3,
    "production-ready": 3,
    "enterprise":       2,
    "thread-safe":      2,
    "concurrent":       2,
    "lru":              2,
    "eviction":         2,
    "ttl":              1,
    "oauth":            2,
    "authentication":   1,
    "authorization":    1,
    "rest api":         1,
    "high availability":3,
    "fault-tolerant":   3,
    "comprehensive":    1,
    "in-depth":         2,
    "detailed":         1,
    "thorough":         1,
    "advanced":         1,
    "end-to-end":       2,
    "across multiple":  2,
    "across 10":        3,
    "market data":      2,
    "multiple industries": 3,
    "scalable":         2,
    "architecture":     2,
}


def _count_tokens(text: str) -> int:
    """Accurate token count via tiktoken; falls back to word-based estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text.split()) * 4 // 3


def _detect_task_type(lower: str) -> tuple[TaskType, float]:
    scores: dict[TaskType, float] = {t: 0.0 for t in TaskType}

    for task_type, groups in _TASK_KEYWORDS.items():
        for kw in groups.get("primary", []):
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                scores[task_type] += 3.0
        for kw in groups.get("secondary", []):
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                scores[task_type] += 1.0

    # Apply negative keywords
    for task_type, neg_kws in _NEGATIVE_KEYWORDS.items():
        for kw in neg_kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                scores[task_type] -= 2.0

    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return TaskType.DOC_CREATION, 0.3

    total = sum(v for v in scores.values() if v > 0) or 1
    confidence = min(scores[best] / total + 0.2, 1.0)
    return best, confidence


def _detect_complexity(lower: str, tokens: int) -> TaskComplexity:
    # Sum escalator weights
    escalator_weight = sum(
        w for kw, w in _ESCALATORS.items() if kw in lower
    )

    # Token-based baseline
    if tokens > 15000:
        base = TaskComplexity.RESEARCH
    elif tokens > 5000:
        base = TaskComplexity.COMPLEX
    elif tokens > 500:
        base = TaskComplexity.STANDARD
    else:
        base = TaskComplexity.SIMPLE

    # Escalate by weight
    _levels = [TaskComplexity.SIMPLE, TaskComplexity.STANDARD,
               TaskComplexity.COMPLEX, TaskComplexity.RESEARCH]
    idx = _levels.index(base)

    if escalator_weight >= 5:
        idx = min(idx + 3, 3)   # jump to RESEARCH for very complex tasks
    elif escalator_weight >= 3:
        idx = min(idx + 2, 3)   # jump to COMPLEX
    elif escalator_weight >= 1:
        idx = min(idx + 1, 3)   # push up one level (e.g. SIMPLE → STANDARD)

    complexity = _levels[idx]

    # De-escalate if simplicity keywords present
    de_hits = sum(1 for kw in _DEESCALATORS if kw in lower)
    if de_hits >= 1 and idx > 0:
        complexity = _levels[idx - 1]

    return complexity


def classify_layer1(task: str) -> tuple[TaskType, TaskComplexity, ModelTier, float, str]:
    if not task or not task.strip():
        raise ClassificationError("Layer 1 received an empty task string.")

    t0 = time.perf_counter()
    lower = task.lower()
    tokens = _count_tokens(task)
    task_type, confidence = _detect_task_type(lower)
    complexity = _detect_complexity(lower, tokens)
    tier = TIER_MATRIX.get((task_type, complexity), ModelTier.MEDIUM)
    elapsed = (time.perf_counter() - t0) * 1000

    reason = (
        f"layer1 | type={task_type.value} complexity={complexity.value} "
        f"tokens={tokens} tier={tier.value} ({elapsed:.1f}ms)"
    )
    logger.debug(reason)
    return task_type, complexity, tier, confidence, reason
