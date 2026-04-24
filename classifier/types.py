from dataclasses import dataclass, field
from enum import Enum


class ModelTier(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class TaskType(Enum):
    REASONING     = "reasoning"
    THINKING      = "thinking"
    ANALYZING     = "analyzing"
    CODE_CREATION = "code_creation"
    DOC_CREATION  = "doc_creation"
    TRANSLATION   = "translation"   # translate, convert language, localize
    MATH          = "math"          # calculate, solve, equation, integral
    CONVERSATION  = "conversation"  # hello, thanks, casual chat → always LOW


class TaskComplexity(Enum):
    SIMPLE   = "simple"    # < 500 tokens, single question
    STANDARD = "standard"  # 500-5K tokens, moderate depth
    COMPLEX  = "complex"   # 5K-15K tokens, multi-step
    RESEARCH = "research"  # > 15K tokens, comprehensive


@dataclass
class ClassificationDecision:
    model_name:  str
    tier:        ModelTier
    task_type:   TaskType
    complexity:  TaskComplexity
    reasoning:   str
    confidence:  float
    provider:    str
    layer_used:  str   = "layer1"  # which layer produced this decision
    latency_ms:  float = 0.0       # total classification time
