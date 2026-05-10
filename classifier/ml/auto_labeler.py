"""Auto-labeler — Snorkel-style weak supervision over decision ⨝ outcome data.

Each labeling function (LF) returns a partial `Label` (any field can be None
when the LF doesn't have signal for it). The aggregator weighted-votes
across LFs to produce one (task_type, complexity, confidence) per
decision_id. Rows below `min_confidence` are dropped.

Output schema matches what `train_head.py` accepts:

    {"task": "...", "task_type": "...", "complexity": "...",
     "_label_confidence": 0.83, "_decision_id": "abc..."}

Usage:

    from classifier.ml.auto_labeler import AutoLabeler

    labeler = AutoLabeler(min_confidence=0.7)
    rows = labeler.run(since="2026-04-01T00:00:00")
    # then write `rows` to JSONL and feed to train_head.

CLI:
    dmr relabel --since 30d --min-confidence 0.7 --out labeled.jsonl
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Label record ─────────────────────────────────────────────────────────────


@dataclass
class Label:
    """Partial label emitted by one labeling function.

    Any of `task_type` / `complexity` may be None when the LF only votes on
    the dimension it has signal for. `confidence` is the LF's self-rated
    certainty in [0, 1] — used for weighted majority aggregation.
    """

    task_type: str | None = None
    complexity: str | None = None
    confidence: float = 0.0
    reason: str = ""


LabelingFunction = Callable[[dict, dict], Label | None]


# ── 8 built-in labeling functions ────────────────────────────────────────────


def lf_short_short(decision: dict, outcome: dict) -> Label | None:
    """Short prompt + short answer + cheap model accepted → simple/conversation."""
    task_len = decision.get("task_length", 0) or len(decision.get("task_preview", ""))
    tokens_out = outcome.get("tokens_out", 0)
    user_retried = outcome.get("user_retried", False)
    if task_len < 60 and tokens_out and tokens_out < 30 and not user_retried:
        return Label(complexity="simple", confidence=0.85, reason="short prompt+short answer, no retry")
    return None


def lf_user_escalated(decision: dict, outcome: dict) -> Label | None:
    """User manually chose a bigger model → original was underspec'd → reasoning/complex."""
    if outcome.get("user_escalated_model"):
        return Label(
            task_type="reasoning",
            complexity="complex",
            confidence=0.95,
            reason="user escalated to bigger model",
        )
    return None


def lf_high_output_ratio(decision: dict, outcome: dict) -> Label | None:
    """tokens_out / tokens_in > 5 → generative task → doc_creation/standard."""
    if outcome.get("tokens_estimated", False):
        return None  # don't trust heuristic counts
    tin = outcome.get("tokens_in", 0) or 0
    tout = outcome.get("tokens_out", 0) or 0
    if tin >= 10 and tout / max(tin, 1) > 5:
        # Sustained generation → likely doc_creation or code_creation
        return Label(
            task_type="doc_creation",
            complexity="standard",
            confidence=0.7,
            reason=f"output/input ratio {tout / tin:.1f}",
        )
    return None


def lf_thumbs_down(decision: dict, outcome: dict) -> Label | None:
    """User downvoted → tier was too low → bump complexity."""
    if outcome.get("user_feedback") == "down":
        return Label(complexity="complex", confidence=0.8, reason="user thumbs-down → bump complexity")
    return None


def lf_thumbs_up_trust_layer1(decision: dict, outcome: dict) -> Label | None:
    """User thumbs-up + L1 was confident → trust L1's labels."""
    if (
        outcome.get("user_feedback") == "up"
        and decision.get("layer") == "layer1"
        and decision.get("confidence", 0) > 0.85
    ):
        return Label(
            task_type=decision.get("task_type"),
            complexity=decision.get("complexity"),
            confidence=0.9,
            reason="user thumbs-up + L1 high confidence",
        )
    return None


def lf_latency_breach(decision: dict, outcome: dict) -> Label | None:
    """Small model took too long → underspec'd → bump complexity."""
    tier = (decision.get("tier") or "").lower()
    wall = outcome.get("wall_ms", 0) or 0
    if tier == "low" and wall > 5000:
        return Label(complexity="standard", confidence=0.65, reason=f"low tier took {wall:.0f}ms")
    return None


def lf_code_keywords_strong(decision: dict, outcome: dict) -> Label | None:
    """L1 already labeled it code with high confidence + no retry → trust."""
    if (
        decision.get("task_type") == "code_creation"
        and decision.get("confidence", 0) > 0.85
        and not outcome.get("user_retried", False)
        and outcome.get("success", True)
    ):
        return Label(
            task_type="code_creation",
            complexity=decision.get("complexity"),
            confidence=min(0.92, decision.get("confidence", 0.5)),
            reason="L1 high-confidence code + clean outcome",
        )
    return None


def lf_l2_ground_truth(decision: dict, outcome: dict) -> Label | None:
    """L2 ran (LLM classifier) AND outcome was successful → trust L2's labels.

    L2 is the LLM-classifier path; when it ran without retries/escalation,
    its labels are the strongest signal we have for that row.
    """
    if (
        decision.get("layer") == "layer2"
        and decision.get("confidence", 0) > 0.7
        and outcome.get("success", True)
        and not outcome.get("user_retried", False)
        and not outcome.get("user_escalated_model")
    ):
        return Label(
            task_type=decision.get("task_type"),
            complexity=decision.get("complexity"),
            confidence=min(0.88, decision.get("confidence", 0.5) + 0.05),
            reason="L2 ran successfully, no user escalation",
        )
    return None


DEFAULT_LFS: list[LabelingFunction] = [
    lf_short_short,
    lf_user_escalated,
    lf_high_output_ratio,
    lf_thumbs_down,
    lf_thumbs_up_trust_layer1,
    lf_latency_breach,
    lf_code_keywords_strong,
    lf_l2_ground_truth,
]


# ── Aggregator ───────────────────────────────────────────────────────────────


class AutoLabeler:
    """Apply LFs to (decision, outcome) pairs and aggregate via weighted vote.

    Args:
        lfs:            Ordered list of labeling functions. Defaults to DEFAULT_LFS.
        min_confidence: Drop labels whose aggregated confidence is below this.
        skip_cached:    Skip rows where decision was a cache hit (`cached=True`).
        skip_exploration: Skip rows where decision was a random exploration sample.
        require_success: Skip rows where outcome.success is False (default True).
    """

    def __init__(
        self,
        lfs: list[LabelingFunction] | None = None,
        *,
        min_confidence: float = 0.7,
        skip_cached: bool = True,
        skip_exploration: bool = True,
        require_success: bool = True,
    ) -> None:
        self.lfs = list(lfs if lfs is not None else DEFAULT_LFS)
        self.min_confidence = float(min_confidence)
        self.skip_cached = skip_cached
        self.skip_exploration = skip_exploration
        self.require_success = require_success

        self._stats: dict[str, int] = defaultdict(int)

    # ── Public API ───────────────────────────────────────────────────────────

    def label_one(self, decision: dict, outcome: dict) -> dict | None:
        """Apply all LFs to one (decision, outcome) pair. Returns aggregated label
        dict or None if no signal / too low confidence.
        """
        votes: list[Label] = []
        for lf in self.lfs:
            try:
                result = lf(decision, outcome)
            except Exception as exc:
                logger.debug("LF %s raised: %s", getattr(lf, "__name__", "?"), exc)
                continue
            if result is not None:
                votes.append(result)

        if not votes:
            self._stats["no_lf_fired"] += 1
            return None

        aggregated = self._aggregate(votes)
        if aggregated is None:
            self._stats["below_confidence"] += 1
            return None

        self._stats["labeled"] += 1
        return aggregated

    def run(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        decisions: list[dict] | None = None,
        outcomes: list[dict] | None = None,
    ) -> list[dict]:
        """Read logs (or use supplied lists), join, label.

        Returns a list of training rows in `train_head.py` schema:
            {"task": "...", "task_type": "...", "complexity": "...",
             "_label_confidence": 0.83, "_decision_id": "..."}

        The leading underscore on `_label_confidence` and `_decision_id` keeps
        them out of the canonical training fields; downstream tools may use
        them for filtering / debugging.
        """
        from classifier.infra.decision_logger import read_decisions
        from classifier.infra.outcome_logger import join_decisions_outcomes, read_outcomes

        if decisions is None:
            decisions = read_decisions(since=since, until=until)
        if outcomes is None:
            outcomes = read_outcomes(since=since, until=until)

        self._stats.clear()
        self._stats["decisions_read"] = len(decisions)
        self._stats["outcomes_read"] = len(outcomes)

        joined = join_decisions_outcomes(decisions, outcomes)
        self._stats["joined"] = len(joined)

        out: list[dict] = []
        for pair in joined:
            d = pair["decision"]
            o = pair["outcome"]

            if self.skip_cached and d.get("cached"):
                self._stats["skipped_cached"] += 1
                continue
            if self.skip_exploration and d.get("exploration"):
                self._stats["skipped_exploration"] += 1
                continue
            if self.require_success and not o.get("success", True):
                self._stats["skipped_failed"] += 1
                continue

            label = self.label_one(d, o)
            if label is None:
                continue

            task_text = d.get("task_preview") or ""
            if not task_text:
                self._stats["skipped_empty_preview"] += 1
                continue

            out.append(
                {
                    "task": task_text,
                    "task_type": label["task_type"],
                    "complexity": label["complexity"],
                    "_label_confidence": label["confidence"],
                    "_decision_id": d.get("decision_id", ""),
                }
            )

        return out

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ── Aggregation ─────────────────────────────────────────────────────────

    def _aggregate(self, votes: list[Label]) -> dict | None:
        """Weighted majority vote on each dimension independently.

        For task_type and complexity separately:
          - Sum each candidate's confidence across LFs that voted for it.
          - Pick the top.
          - Aggregated confidence = top_score / sum_all_scores.
        Final confidence = mean of the two dimensions' shares.
        """
        tt_scores: dict[str, float] = defaultdict(float)
        cx_scores: dict[str, float] = defaultdict(float)

        for v in votes:
            if v.task_type:
                tt_scores[v.task_type] += v.confidence
            if v.complexity:
                cx_scores[v.complexity] += v.confidence

        if not tt_scores or not cx_scores:
            return None

        top_tt, tt_top = max(tt_scores.items(), key=lambda kv: kv[1])
        top_cx, cx_top = max(cx_scores.items(), key=lambda kv: kv[1])

        tt_total = sum(tt_scores.values())
        cx_total = sum(cx_scores.values())
        tt_share = tt_top / tt_total if tt_total > 0 else 0.0
        cx_share = cx_top / cx_total if cx_total > 0 else 0.0

        agg_conf = (tt_share + cx_share) / 2

        if agg_conf < self.min_confidence:
            return None

        return {
            "task_type": top_tt,
            "complexity": top_cx,
            "confidence": round(agg_conf, 3),
        }
