from enum import Enum
from dataclasses import dataclass


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


class TaskComplexity(Enum):
    SIMPLE   = "simple"
    STANDARD = "standard"
    COMPLEX  = "complex"
    RESEARCH = "research"


@dataclass
class ClassificationDecision:
    model_name:  str
    tier:        ModelTier
    task_type:   TaskType
    complexity:  TaskComplexity
    reasoning:   str
    confidence:  float
    provider:    str
