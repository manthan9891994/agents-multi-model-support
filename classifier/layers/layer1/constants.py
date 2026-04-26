from classifier.core.types import TaskType, TaskComplexity, ModelTier

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
                    "roadmap", "workflow",
                    "propose", "outline", "blueprint", "spec", "rfc"],
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
                    "fix bug", "implement", "debug", "refactor",
                    "profile", "benchmark", "lint", "deploy", "dockerize",
                    "migrate", "scaffold", "boilerplate"],
        "secondary": ["function", "algorithm", "optimize", "class", "code",
                      "develop", "programming", "build"],
    },
    TaskType.DOC_CREATION: {
        "primary": ["create documentation", "write a summary", "write a report",
                    "write a tutorial", "write a guide", "write a readme",
                    "write documentation", "write the",
                    "paraphrase", "rewrite", "rephrase", "proofread", "edit this"],
        "secondary": ["document", "readme", "guide", "manual", "summarize",
                      "describe", "explain", "comment"],
    },
}

_NEGATIVE_KEYWORDS: dict[TaskType, list[str]] = {
    TaskType.CODE_CREATION: ["explain", "what is", "tell me about", "how does",
                              "describe what"],
    TaskType.REASONING:     ["calculate", "compute", "solve for"],
    TaskType.MATH:          ["write", "implement", "create", "design"],
}

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

_DEESCALATORS = {
    "simple", "basic", "quick", "brief", "one-line", "trivial",
    "short", "easy", "beginner", "just a", "small", "example only",
    "in one sentence", "in a few words", "one-liner", "tldr", "just the gist",
}

_ALGORITHM_NAMES = {
    "raft", "paxos", "b-tree", "bloom filter", "consistent hashing",
    "red-black tree", "avl tree", "dijkstra", "byzantine fault",
    "two-phase commit", "saga pattern", "cqrs", "event sourcing",
    "merkle tree", "lsm tree", "skip list", "chord protocol",
    "vector clock", "crdt",
}

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

_FORMAT_REQUESTS = {
    "return json", "as json", "in json format", "json output",
    "as a table", "in table format", "formatted as",
    "in bullet points", "as a list", "in yaml", "as csv",
    "as markdown", "in markdown", "as xml",
}

_TASK_TYPE_TIER_WEIGHT: dict[TaskType, int] = {
    TaskType.CONVERSATION:  0,
    TaskType.DOC_CREATION:  1,
    TaskType.TRANSLATION:   2,
    TaskType.MATH:          3,
    TaskType.ANALYZING:     3,
    TaskType.CODE_CREATION: 4,
    TaskType.REASONING:     5,
    TaskType.THINKING:      5,
    TaskType.MULTIMODAL:    5,
}

_TIER_ORDER = [ModelTier.LOW, ModelTier.MEDIUM, ModelTier.HIGH]
_COMPLEXITY_LEVELS = [
    TaskComplexity.SIMPLE, TaskComplexity.STANDARD,
    TaskComplexity.COMPLEX, TaskComplexity.RESEARCH,
]
_LOW_TIER_CONTEXT_TOKENS = 8_192
