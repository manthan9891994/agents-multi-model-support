"""Budget-aware cost tracking. Records token usage per model and signals downgrade
when monthly spend exceeds the configured threshold."""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

COST_PER_1M_TOKENS: dict[str, float] = {
    "gemini-2.5-flash-lite":  0.05,
    "gemini-2.5-flash":       0.25,
    "gemini-2.5-pro":         2.50,
    "claude-haiku-4-5-20251001": 0.40,
    "claude-sonnet-4-6":         3.00,
    "claude-opus-4-7":           15.00,
    "gpt-4o-mini":  0.15,
    "gpt-4o":       2.50,
    "gpt-4-turbo":  10.00,
}


@dataclass
class UsageRecord:
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    timestamp:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CostTracker:
    def __init__(self, monthly_budget_usd: float = 1000.0):
        self.monthly_budget = monthly_budget_usd
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int = 500) -> float:
        total_tokens = input_tokens + output_tokens
        rate = COST_PER_1M_TOKENS.get(model, 0.25)
        cost = (total_tokens / 1_000_000) * rate
        with self._lock:
            self._records.append(
                UsageRecord(model=model, input_tokens=input_tokens,
                            output_tokens=output_tokens, cost_usd=cost)
            )
        logger.debug("Recorded %.6f USD for %s (%d tokens)", cost, model, total_tokens)
        return cost

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self._records)

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
            for r in self._records:
                by_model[r.model] = by_model.get(r.model, 0.0) + r.cost_usd
        return {
            "total_cost_usd":   round(self.total_cost, 6),
            "budget_usd":       self.monthly_budget,
            "budget_remaining": round(self.budget_remaining, 6),
            "utilization_pct":  round(self.budget_utilization * 100, 2),
            "by_model":         {k: round(v, 6) for k, v in by_model.items()},
            "total_calls":      len(self._records),
        }


cost_tracker = CostTracker()
