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

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (enums → string values)."""
        return {
            "model_name":      self.model_name,
            "tier":            self.tier.value,
            "task_type":       self.task_type.value,
            "complexity":      self.complexity.value,
            "reasoning":       self.reasoning,
            "confidence":      self.confidence,
            "provider":        self.provider,
            "layer_used":      self.layer_used,
            "latency_ms":      self.latency_ms,
            "compliance_flag": self.compliance_flag,
            "disagreement":    self.disagreement,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationDecision":
        """Deserialise from a dict (string values → enums)."""
        return cls(
            model_name      = data["model_name"],
            tier            = ModelTier(data["tier"]),
            task_type       = TaskType(data["task_type"]),
            complexity      = TaskComplexity(data["complexity"]),
            reasoning       = data.get("reasoning", ""),
            confidence      = float(data.get("confidence", 0.0)),
            provider        = data.get("provider", "google"),
            layer_used      = data.get("layer_used", "layer1"),
            latency_ms      = float(data.get("latency_ms", 0.0)),
            compliance_flag = bool(data.get("compliance_flag", False)),
            disagreement    = bool(data.get("disagreement", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "ClassificationDecision":
        import json
        return cls.from_dict(json.loads(raw))


@dataclass
class ContextSignals:
    """Signals from the full LLM request for agent mid-flight tier adjustment."""
    total_context_tokens: int  = 0
    call_number:          int  = 1
    has_error:            bool = False
    last_role:            str  = "user"
    has_multimodal:       bool = False  # inline_data or file_data parts in request
    available_tools:      int  = 0      # number of tools exposed to the agent
