import logging
import re

from classifier.exceptions import ClassificationError
from classifier.types import TaskType, TaskComplexity, ModelTier
from classifier.registry import TIER_MATRIX

logger = logging.getLogger(__name__)

_TASK_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.REASONING: [
        "analyze", "compare", "evaluate", "assess", "debate", "interpret",
        "distinguish", "validate", "argument", "logic", "contrast",
        "pros and cons", "trade-off", "tradeoff", "why", "should i",
        "which is better", "difference between", "versus", "vs",
    ],
    TaskType.THINKING: [
        "plan", "strategy", "brainstorm", "explore", "design", "ideate",
        "organize", "structure", "approach", "workflow", "process",
        "architect", "system design", "how should", "what would", "roadmap",
    ],
    TaskType.ANALYZING: [
        "data", "statistics", "pattern", "trend", "insight", "metric",
        "aggregate", "breakdown", "distribution", "correlation", "anomaly",
        "benchmark", "measure", "chart", "graph", "dataset", "seasonal",
    ],
    TaskType.CODE_CREATION: [
        "code", "function", "implement", "debug", "refactor", "optimize",
        "algorithm", "class", "fix bug", "write code", "programming",
        "bug", "develop", "create a function", "def ", "api endpoint",
        "unit test", "write a test",
    ],
    TaskType.DOC_CREATION: [
        "document", "documentation", "readme", "guide", "manual",
        "tutorial", "comment", "describe", "summarize", "summary",
        "report", "write a", "write the",
    ],
}

# Keywords that push complexity up — one hit → STANDARD, two+ → COMPLEX/RESEARCH
_ESCALATORS = {
    "distributed", "microservices", "multi-step", "architecture",
    "comprehensive", "enterprise", "production", "scalable",
    "end-to-end", "full system", "advanced", "thread-safe",
    "concurrent", "high availability", "fault-tolerant",
    "across multiple", "multiple industries", "in-depth", "detailed",
    "thorough", "lru", "eviction", "ttl",
    "rest api", "oauth", "authentication", "authorization",
    "across 10", "50 data", "market data", "all industries",
}


def _matches(keyword: str, text: str) -> bool:
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return bool(re.search(pattern, text))


def _estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3


def _detect_task_type(lower: str) -> tuple[TaskType, float]:
    scores: dict[TaskType, int] = {t: 0 for t in TaskType}
    for task_type, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if _matches(kw, lower):
                scores[task_type] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return TaskType.DOC_CREATION, 0.3

    total = sum(scores.values())
    confidence = min(scores[best] / total + 0.3, 1.0)
    return best, confidence


def _detect_complexity(lower: str, tokens: int) -> TaskComplexity:
    hits = sum(1 for kw in _ESCALATORS if kw in lower)

    if tokens > 15000 or hits >= 3:
        return TaskComplexity.RESEARCH
    if tokens > 5000 or hits >= 2:
        return TaskComplexity.COMPLEX
    if tokens > 500 or hits >= 1:
        return TaskComplexity.STANDARD
    return TaskComplexity.SIMPLE


def classify_layer1(task: str) -> tuple[TaskType, TaskComplexity, ModelTier, float, str]:
    if not task or not task.strip():
        raise ClassificationError("Layer 1 received an empty task string.")

    lower = task.lower()
    tokens = _estimate_tokens(task)
    task_type, confidence = _detect_task_type(lower)
    complexity = _detect_complexity(lower, tokens)
    tier = TIER_MATRIX.get((task_type, complexity), ModelTier.MEDIUM)
    reason = (
        f"layer1 | type={task_type.value} complexity={complexity.value} "
        f"tokens={tokens} tier={tier.value}"
    )
    logger.debug(reason)
    return task_type, complexity, tier, confidence, reason
