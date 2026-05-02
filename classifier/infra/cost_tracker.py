"""Budget-aware cost tracking. Records token usage per model and signals downgrade
when monthly spend exceeds the configured threshold."""
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Cost table populated from YAML at import time. Empty by default.
# Use register_model_cost() to add or update.
COST_TABLE: dict[str, dict[str, float]] = {}

# Used when a model has no registered cost. Conservative midpoint so unknown
# models don't exhaust the budget instantly. Override with register_model_cost.
_DEFAULT_COST = {"input": 0.25, "output": 0.75}


def register_model_cost(
    model: str,
    *,
    input_per_1m: float,
    output_per_1m: float,
) -> None:
    """Register or override the cost rates for a model.

    Example:
        register_model_cost("mistral-large-2", input_per_1m=2.0, output_per_1m=6.0)
    """
    COST_TABLE[model] = {"input": float(input_per_1m), "output": float(output_per_1m)}


def get_model_cost(model: str) -> dict[str, float]:
    return COST_TABLE.get(model, _DEFAULT_COST)


# Back-compat alias — keeps any external imports working.
def _legacy_per_1m(model: str) -> float:
    """Returns the average of input+output rate for legacy callers."""
    c = get_model_cost(model)
    return (c["input"] + c["output"]) / 2


# Module-level dict that mimics the old flat shape for back-compat
class _LegacyCostMap(dict):
    def __getitem__(self, key):
        return _legacy_per_1m(key)
    def get(self, key, default=None):
        if key in COST_TABLE:
            return _legacy_per_1m(key)
        return default
    def __contains__(self, key):
        return key in COST_TABLE


COST_PER_1M_TOKENS = _LegacyCostMap()   # back-compat shim


def _is_test_mode() -> bool:
    return os.environ.get("CLASSIFIER_TEST_MODE", "").lower() in ("1", "true", "yes")


@dataclass
class UsageRecord:
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    category:      str  = "main"
    timestamp:     str  = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostTracker:
    def __init__(self, monthly_budget_usd: float = 1000.0):
        self.monthly_budget = monthly_budget_usd
        self._records: list[UsageRecord] = []
        self._category_budgets: dict[str, float] = {}
        self._lock = threading.Lock()

    def set_category_budget(self, category: str, budget: float) -> None:
        """Set a per-category spending limit (e.g., 'layer2' capped at $50)."""
        self._category_budgets[category] = budget

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 500,
        category: str = "main",
    ) -> float:
        if _is_test_mode():
            return 0.0
        rates = get_model_cost(model)
        cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
        with self._lock:
            self._records.append(
                UsageRecord(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    category=category,
                )
            )
        logger.debug("Recorded %.6f USD for %s (%d/%d tokens) [%s]", cost, model, input_tokens, output_tokens, category)
        return cost

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def cost_for_category(self, category: str) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records if r.category == category)

    def is_exhausted_for(self, category: str) -> bool:
        """Return True when a category's budget is set and fully spent."""
        cat_budget = self._category_budgets.get(category)
        if cat_budget is None or cat_budget <= 0:
            return False
        return self.cost_for_category(category) >= cat_budget

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.monthly_budget - self.total_cost)

    @property
    def budget_utilization(self) -> float:
        return self.total_cost / self.monthly_budget if self.monthly_budget > 0 else 0.0

    def should_downgrade(self) -> bool:
        over = self.budget_utilization > 0.80
        if over:
            logger.warning(
                "Budget utilization %.0f%% — downgrading to MEDIUM tier max.",
                self.budget_utilization * 100,
            )
        return over

    def is_exhausted(self) -> bool:
        exhausted = self.budget_utilization >= 1.0
        if exhausted:
            logger.error("Monthly budget exhausted — forcing LOW tier.")
        return exhausted

    @property
    def summary(self) -> dict:
        with self._lock:
            by_model: dict[str, float] = {}
            by_category: dict[str, float] = {}
            for r in self._records:
                by_model[r.model]        = by_model.get(r.model, 0.0)        + r.cost_usd
                by_category[r.category]  = by_category.get(r.category, 0.0)  + r.cost_usd
        return {
            "total_cost_usd":   round(self.total_cost, 6),
            "budget_usd":       self.monthly_budget,
            "budget_remaining": round(self.budget_remaining, 6),
            "utilization_pct":  round(self.budget_utilization * 100, 2),
            "by_model":         {k: round(v, 6) for k, v in by_model.items()},
            "by_category":      {k: round(v, 6) for k, v in by_category.items()},
            "total_calls":      len(self._records),
        }


cost_tracker = CostTracker()
