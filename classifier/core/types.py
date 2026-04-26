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
    MULTIMODAL    = "multimodal"    # image/audio/vision tasks


class TaskComplexity(Enum):
    SIMPLE   = "simple"    # < 500 tokens, single question
    STANDARD = "standard"  # 500-5K tokens, moderate depth
    COMPLEX  = "complex"   # 5K-15K tokens, multi-step
    RESEARCH = "research"  # > 15K tokens, comprehensive


@dataclass
class ClassificationDecision:
    model_name:      str
    tier:            ModelTier
    task_type:       TaskType
    complexity:      TaskComplexity
    reasoning:       str
    confidence:      float
    provider:        str
    layer_used:      str   = "layer1"
    latency_ms:      float = 0.0
    compliance_flag: bool  = False  # PII/PHI/secret detected in task
    disagreement:    bool  = False  # L1 and L2 disagreed on classification


@dataclass
class ContextSignals:
    """Signals from the full LLM request for agent mid-flight tier adjustment."""
    total_context_tokens: int  = 0
    call_number:          int  = 1
    has_error:            bool = False
    last_role:            str  = "user"
    has_multimodal:       bool = False  # inline_data or file_data parts in request
    available_tools:      int  = 0      # number of tools exposed to the agent
