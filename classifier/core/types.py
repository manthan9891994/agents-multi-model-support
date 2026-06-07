from dataclasses import dataclass, field
from enum import Enum


class _OpenEnumMeta(type(Enum)):
    """Metaclass that allows extending an Enum at runtime via cls._add_member_().

    Used to make TaskType and TaskComplexity user-extensible without breaking
    existing `TaskType.REASONING` member access.
    """

    def __call__(cls, value):
        # Look up dynamic registry first
        dyn = getattr(cls, "_dynamic_members_", {})
        if value in dyn:
            return dyn[value]
        return super().__call__(value)


def _add_member(enum_cls, name: str, value: str):
    """Register a new member on an Enum-like class at runtime.

    Returns the member-like object. Stored in `enum_cls._dynamic_members_`
    so `EnumCls(value)` finds it.
    """
    dyn = getattr(enum_cls, "_dynamic_members_", None)
    if dyn is None:
        dyn = {}
        enum_cls._dynamic_members_ = dyn
    if value in dyn:
        return dyn[value]

    # Build a lightweight member object that quacks like an Enum member
    class _DynamicMember:
        __slots__ = ("name", "value")

        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __repr__(self):
            return f"<{enum_cls.__name__}.{self.name}: {self.value!r}>"

        def __eq__(self, other):
            return getattr(other, "value", None) == self.value

        def __hash__(self):
            return hash((enum_cls.__name__, self.value))

    member = _DynamicMember(name, value)
    dyn[value] = member
    setattr(enum_cls, name, member)
    return member


class ModelTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Configurable tier ordering. Updated by set_tier_levels() to support 4+ tiers.
# Internal modules import _TIER_ORDER instead of hardcoding [LOW, MEDIUM, HIGH].
_TIER_ORDER: list = [ModelTier.LOW, ModelTier.MEDIUM, ModelTier.HIGH]


class CallRole(str, Enum):
    """How a model call is used inside an agent loop (drives capability gating)."""

    CONVERSATIONAL = "conversational"  # greeting/chit-chat, no tools needed
    ORCHESTRATION = "orchestration"  # pick/route to a sub-agent or tool
    TOOL_CALL = "tool_call"  # drive a tool (needs reliable tool-calling)
    SYNTHESIS = "synthesis"  # write the user-facing answer from gathered data


class RoutingScope(str, Enum):
    """Granularity at which a routing decision is reused (stickiness)."""

    CALL = "call"  # decide every call (default; backward compatible)
    TURN = "turn"  # one decision per user turn (whole tool loop)
    AGENT = "agent"  # one decision per agent
    CONVERSATION = "conversation"


class Effort(str, Enum):
    """Reasoning/thinking budget, applied by adapters to provider-native controls."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


def set_tier_levels(names: list[str]) -> list:
    """Replace the global tier ordering with a custom list of names.

    Existing LOW/MEDIUM/HIGH constants stay valid. New tiers are dynamically
    added as enum-like members.

    Example:
        set_tier_levels(["free", "cheap", "standard", "premium", "frontier"])

        router = Router(model_registry={
            "openai": {
                "free": "gpt-4o-mini", "cheap": "gpt-4o-mini",
                "standard": "gpt-4o", "premium": "gpt-4-turbo",
                "frontier": "o1-pro",
            },
        })
    """
    global _TIER_ORDER
    new_order: list = []
    for n in names:
        try:
            new_order.append(ModelTier(n))
        except ValueError:
            new_order.append(_add_member(ModelTier, n.upper(), n))
    _TIER_ORDER = new_order
    return new_order


def list_tier_levels() -> list[str]:
    return [t.value for t in _TIER_ORDER]


class TaskType(Enum):
    REASONING = "reasoning"
    THINKING = "thinking"
    ANALYZING = "analyzing"
    CODE_CREATION = "code_creation"
    DOC_CREATION = "doc_creation"
    TRANSLATION = "translation"
    MATH = "math"
    CONVERSATION = "conversation"
    MULTIMODAL = "multimodal"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    RESEARCH = "research"


def task_type_for(value: str):
    """Look up a TaskType by string value, including dynamically registered ones.

    Use this instead of `TaskType(value)` when value may be a custom registered type.
    """
    dyn = getattr(TaskType, "_dynamic_members_", {})
    if value in dyn:
        return dyn[value]
    return TaskType(value)


def complexity_for(value: str):
    """Look up a TaskComplexity by string value, including dynamic ones."""
    dyn = getattr(TaskComplexity, "_dynamic_members_", {})
    if value in dyn:
        return dyn[value]
    return TaskComplexity(value)


def register_task_type(value: str, *, name: str | None = None) -> "TaskType":
    """Register a new task type at runtime. Returns a member-like object.

    The new type can be used wherever TaskType members are expected. The
    user is responsible for adding tier matrix entries for any new types
    they want routed.

    Example:
        ct = register_task_type("clinical_note")
        # Then:
        from classifier import TIER_MATRIX, TaskComplexity, ModelTier
        TIER_MATRIX[(ct, TaskComplexity.STANDARD)] = ModelTier.HIGH
    """
    return _add_member(TaskType, name or value.upper(), value)


def register_complexity(value: str, *, name: str | None = None) -> "TaskComplexity":
    """Register a new complexity level at runtime."""
    return _add_member(TaskComplexity, name or value.upper(), value)


def _new_decision_id() -> str:
    """UUID4-based decision id used to join decisions ⨝ outcomes."""
    import uuid

    return uuid.uuid4().hex[:16]


@dataclass
class ClassificationDecision:
    model_name: str
    tier: ModelTier
    task_type: TaskType
    complexity: TaskComplexity
    reasoning: str
    confidence: float
    provider: str
    layer_used: str = "layer1"
    latency_ms: float = 0.0
    compliance_flag: bool = False  # PII/PHI/secret detected in task
    disagreement: bool = False  # L1 and L2 disagreed on classification
    decision_id: str = field(default_factory=_new_decision_id)
    exploration: bool = False  # set True by Explorer when this call is a random sample
    cached: bool = False  # True when returned from in-process or pluggable cache
    cached_from: str = ""  # decision_id of the original (uncached) decision
    sticky: bool = False  # reused from a prior decision within the same routing scope
    call_role: str = "synthesis"  # CallRole value — how this call is used in an agent loop
    effort: str = "none"  # Effort value — reasoning/thinking budget for this call
    cache_state: str = "cold"  # "warm" if the provider prompt cache is likely hot

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (enums → string values)."""
        return {
            "model_name": self.model_name,
            "tier": self.tier.value,
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "provider": self.provider,
            "layer_used": self.layer_used,
            "latency_ms": self.latency_ms,
            "compliance_flag": self.compliance_flag,
            "disagreement": self.disagreement,
            "decision_id": self.decision_id,
            "exploration": self.exploration,
            "cached": self.cached,
            "cached_from": self.cached_from,
            "sticky": self.sticky,
            "call_role": self.call_role,
            "effort": self.effort,
            "cache_state": self.cache_state,
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationDecision":
        """Deserialise from a dict (string values → enums).

        Emits a logger warning if `decision_id` is missing — a fresh one is
        minted but this breaks decision ⨝ outcome joins for the row.
        """
        import logging as _log

        if not data.get("decision_id"):
            _log.getLogger(__name__).warning(
                "ClassificationDecision.from_dict: missing decision_id — "
                "minting a new one. Joins against outcome logs will fail for this row."
            )
        return cls(
            model_name=data["model_name"],
            tier=ModelTier(data["tier"]),
            task_type=TaskType(data["task_type"]),
            complexity=TaskComplexity(data["complexity"]),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
            provider=data.get("provider", "google"),
            layer_used=data.get("layer_used", "layer1"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            compliance_flag=bool(data.get("compliance_flag", False)),
            disagreement=bool(data.get("disagreement", False)),
            decision_id=data.get("decision_id") or _new_decision_id(),
            exploration=bool(data.get("exploration", False)),
            cached=bool(data.get("cached", False)),
            cached_from=data.get("cached_from", "") or "",
            sticky=bool(data.get("sticky", False)),
            call_role=data.get("call_role", "synthesis") or "synthesis",
            effort=data.get("effort", "none") or "none",
            cache_state=data.get("cache_state", "cold") or "cold",
        )

    @classmethod
    def from_json(cls, raw: str) -> "ClassificationDecision":
        import json

        return cls.from_dict(json.loads(raw))


@dataclass
class ContextSignals:
    """Signals from the full LLM request for agent mid-flight tier adjustment."""

    total_context_tokens: int = 0
    call_number: int = 1
    has_error: bool = False
    last_role: str = "user"
    has_multimodal: bool = False  # inline_data or file_data parts in request
    available_tools: int = 0  # number of tools exposed to the agent
    scope_key: str = ""  # turn/agent/conversation id — enables sticky routing
    call_role: str = "synthesis"  # CallRole value — drives capability gating
