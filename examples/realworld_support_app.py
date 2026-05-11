"""Real-world usecase: customer-support assistant routing 50 mixed queries.

Scenario: You're building a support bot. Users send a mix of questions —
some trivial ("what time do you open?"), some complex ("explain why my
invoice doesn't match the contract terms for clause 7.2"). Sending every
query to GPT-4 / Claude Opus costs $$ and adds latency.

This script simulates 50 real queries, routes each one through the cascade,
fake-calls the chosen LLM, reports outcomes, then runs SQL analytics over
the SQLite telemetry store.

Run:
    python examples/realworld_support_app.py

Expected output:
  - 50 routing decisions, mixed tiers
  - Per-tier cost breakdown
  - Latency p50/p95
  - Top 5 most expensive queries
  - Estimate: what we'd have paid sending everything to the highest tier
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from classifier import MultiLoggerBackend, OutcomeRecord, Router, StdoutLoggerBackend
from classifier.infra.outcome_logger import log_outcome
from custom_backends.sqlite_backend import SQLiteBackend


# Quiet most logs, surface only key events
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
for n in ("urllib3", "httpcore", "httpx", "filelock", "sentence_transformers", "classifier"):
    logging.getLogger(n).setLevel(logging.WARNING)


# ── Realistic mixed traffic ──────────────────────────────────────────────────

SUPPORT_QUERIES = [
    # FAQ-style (should route LOW)
    "what time do you open on weekends",
    "how do I reset my password",
    "where can I see my order history",
    "do you ship internationally",
    "what is your return policy",
    "is there a mobile app",
    "how do I contact a human agent",
    "what payment methods do you accept",
    "do you offer student discounts",
    "is my data encrypted",
    "what is the shipping cost to Canada",
    "how do I cancel my subscription",
    "where is my package",
    "can I change my delivery address",
    "what is your phone number",
    # Code/translation tasks (LOW)
    "translate 'thank you' into German",
    "convert 100 USD to EUR",
    "summarize this in one sentence: We are pleased to announce...",
    "what is 15% of 240",
    "format this date 2026-05-10 as MM/DD/YYYY",
    # Medium complexity (should route MEDIUM)
    "compare the warranty terms between your basic and premium plans",
    "explain why my invoice shows two charges this month",
    "I was double-charged on March 15, can you walk me through why",
    "help me decide between the family plan and two individual plans",
    "analyze my last 6 months of orders and tell me my top categories",
    "compare the cancellation fees if I switch from annual to monthly",
    "explain the difference between gross and net in my last bill",
    "why is my shipping cost higher than what the website quoted",
    "what's the most cost-effective plan for a team of 12",
    "should I bundle my insurance with my subscription",
    # High complexity (should route HIGH)
    "explain why my invoice doesn't match the contract terms in clause 7.2 about quarterly billing",
    "design a workflow for handling refund disputes when the customer claims fraud and the merchant claims it was authorized",
    "I have a complex tax situation — non-resident in Germany but employed by a US LLC, how does that affect my subscription VAT",
    "audit my last 24 months of usage and recommend a custom enterprise plan with SLA recommendations",
    "I'm building an integration — explain the rate limits, retry strategies, idempotency keys, and what counts as a partial failure",
    "write a step-by-step migration plan from your v1 API to v2 for a high-volume e-commerce site with 50M records",
    "interpret these conflicting clauses in section 12 of the SLA and suggest amendments",
    # Edge cases mixed in
    "hi",
    "?",
    "thanks",
    "ok",
    "please help with my account number 4532-1234-5678-9010 and SSN 123-45-6789",  # PII test
    "my email is user@example.com",  # PII test
    "what",
    "yes",
    "no",
    "what about now",
    "and what if I add a second user",
    "ok thanks bye",
    "wait one more question",
    "actually never mind",
]


# Fake costs for the simulation (USD per token, approximate market rates 2026)
COST_PER_1K_TOKENS = {
    "low": {"in": 0.000_075, "out": 0.000_300},     # cheap models (e.g. gemini-2.5-flash-lite)
    "medium": {"in": 0.000_300, "out": 0.001_200},   # mid (e.g. gemini-2.5-flash)
    "high": {"in": 0.003_000, "out": 0.015_000},     # premium (e.g. gemini-2.5-pro)
}


def fake_llm_call(tier: str, task: str) -> dict:
    """Simulate calling the chosen LLM. Returns realistic tokens/wall_ms."""
    tokens_in = max(4, len(task.split()) * 2)
    # Higher tiers tend to produce longer responses
    tokens_out = {"low": 80, "medium": 250, "high": 600}[tier] + random.randint(-30, 60)
    wall_ms = {"low": 250, "medium": 800, "high": 2400}[tier] + random.randint(-50, 200)
    success = random.random() > 0.02  # 2% failure rate
    rates = COST_PER_1K_TOKENS[tier]
    cost = (tokens_in / 1000.0) * rates["in"] + (tokens_out / 1000.0) * rates["out"]
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_ms": wall_ms,
        "success": success,
        "cost_usd": cost,
    }


def main() -> None:
    print("=" * 72)
    print(" Real-world test: customer-support assistant routing 50 queries")
    print("=" * 72)

    # Clean previous run
    db_path = Path("_support_app.db")
    if db_path.exists():
        db_path.unlink()

    # Set up telemetry: SQLite for persistence + stdout commented out for cleaner output
    sqlite_backend = SQLiteBackend(str(db_path))
    backend = MultiLoggerBackend([sqlite_backend])  # add StdoutLoggerBackend() to also see live

    router = Router(
        decision_logger=backend,
        outcome_logger=backend,
        layer2_enabled=False,
        layer3_enabled=False,
        cache_enabled=True,
    )

    print(f"\nProcessing {len(SUPPORT_QUERIES)} support queries...\n")

    # ── Process queries like a real app ─────────────────────────────────────
    t_start = time.time()
    queries_processed = 0
    decisions = []
    for query in SUPPORT_QUERIES:
        # 1. Router decides which tier to use
        decision = router.classify(query)
        decisions.append(decision)

        # 2. App calls the chosen model
        llm_result = fake_llm_call(decision.tier.value, query)

        # 3. App reports what happened
        router.report_outcome(
            decision_id=decision.decision_id,
            tokens_in=llm_result["tokens_in"],
            tokens_out=llm_result["tokens_out"],
            wall_ms=llm_result["wall_ms"],
            success=llm_result["success"],
            cost_usd=llm_result["cost_usd"],
            user_feedback=random.choice([None, None, None, "up", "down"]),  # 40% feedback
        )
        queries_processed += 1

    wall = time.time() - t_start
    print(f"Processed {queries_processed} queries in {wall:.2f}s ({queries_processed/wall:.0f} qps)")

    # ── Run analytics via plain SQL ────────────────────────────────────────
    print()
    print("=" * 72)
    print(" Analytics (queried from SQLite — no extra tooling needed)")
    print("=" * 72)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Tier distribution
    print("\n1. Tier distribution (cost-tier routing decisions)")
    print("   " + "-" * 50)
    rows = conn.execute(
        "SELECT tier, COUNT(*) AS n FROM events WHERE event_type='decision' GROUP BY tier ORDER BY n DESC"
    ).fetchall()
    total_decisions = sum(r["n"] for r in rows)
    for r in rows:
        pct = r["n"] / total_decisions * 100
        bar = "#" * int(pct / 2)
        print(f"   {r['tier']:8} {r['n']:4d}  {pct:5.1f}%  {bar}")

    # Cost per tier
    print("\n2. Actual cost per tier (USD)")
    print("   " + "-" * 50)
    rows = conn.execute(
        """SELECT d.tier,
                  COUNT(*) AS n_calls,
                  COALESCE(SUM(json_extract(o.payload, '$.tokens_in')), 0)  AS tot_in,
                  COALESCE(SUM(json_extract(o.payload, '$.tokens_out')), 0) AS tot_out,
                  COALESCE(SUM(json_extract(o.payload, '$.cost_usd')), 0)   AS tot_cost
           FROM events d
           JOIN events o
             ON d.decision_id = o.decision_id AND o.event_type='outcome'
           WHERE d.event_type='decision'
           GROUP BY d.tier
           ORDER BY tot_cost DESC"""
    ).fetchall()
    grand_total = 0.0
    for r in rows:
        print(
            f"   {r['tier']:8}  calls={r['n_calls']:3d}  "
            f"tokens={r['tot_in']:5}/{r['tot_out']:5}  cost=${r['tot_cost']:.6f}"
        )
        grand_total += r["tot_cost"]
    print(f"   {'TOTAL':8}                                          ${grand_total:.6f}")

    # Counterfactual: what if we routed everything to HIGH?
    print("\n3. Counterfactual: what if we sent all queries to the HIGH tier?")
    print("   " + "-" * 50)
    rows = conn.execute(
        """SELECT COALESCE(SUM(json_extract(o.payload, '$.tokens_in')), 0)  AS tot_in,
                  COALESCE(SUM(json_extract(o.payload, '$.tokens_out')), 0) AS tot_out
           FROM events o
           WHERE o.event_type='outcome'"""
    ).fetchone()
    high_rates = COST_PER_1K_TOKENS["high"]
    counterfactual = (rows["tot_in"] / 1000.0) * high_rates["in"] + (rows["tot_out"] / 1000.0) * high_rates["out"]
    savings = counterfactual - grand_total
    pct_saved = savings / counterfactual * 100 if counterfactual else 0
    print(f"   All-HIGH cost would be:   ${counterfactual:.6f}")
    print(f"   Actual routed cost:       ${grand_total:.6f}")
    print(f"   Savings:                  ${savings:.6f}  ({pct_saved:.1f}% cheaper)")

    # Latency percentiles
    print("\n4. Latency percentiles (wall time of LLM call)")
    print("   " + "-" * 50)
    latencies = [
        json.loads(row[0])["wall_ms"]
        for row in conn.execute("SELECT payload FROM events WHERE event_type='outcome'")
    ]
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print(f"   p50: {p50:.0f}ms")
    print(f"   p95: {p95:.0f}ms")

    # Success rate
    print("\n5. Success rate")
    print("   " + "-" * 50)
    row = conn.execute(
        "SELECT AVG(CAST(json_extract(payload, '$.success') AS INTEGER)) "
        "FROM events WHERE event_type='outcome'"
    ).fetchone()
    print(f"   {(row[0] or 0) * 100:.1f}%")

    # User feedback distribution
    print("\n6. User feedback")
    print("   " + "-" * 50)
    rows = conn.execute(
        """SELECT json_extract(payload, '$.user_feedback') AS fb, COUNT(*) AS n
           FROM events WHERE event_type='outcome' GROUP BY fb"""
    ).fetchall()
    for r in rows:
        label = r["fb"] or "(none)"
        print(f"   {label:8} {r['n']}")

    # PII compliance flags
    print("\n7. PII / compliance flags raised")
    print("   " + "-" * 50)
    rows = conn.execute(
        """SELECT decision_id, json_extract(payload, '$.task_preview') AS preview
           FROM events
           WHERE event_type='decision' AND json_extract(payload, '$.compliance_flag')=1"""
    ).fetchall()
    if rows:
        for r in rows:
            print(f"   [{r['decision_id'][:8]}] {r['preview'][:60]}...")
    else:
        print("   (none — PII spans were redacted but not flagged as compliance)")

    # Top 5 most expensive
    print("\n8. Top 5 most expensive queries")
    print("   " + "-" * 50)
    rows = conn.execute(
        """SELECT json_extract(d.payload, '$.task_preview') AS preview,
                  d.tier,
                  json_extract(o.payload, '$.cost_usd') AS cost
           FROM events d
           JOIN events o ON d.decision_id = o.decision_id AND o.event_type='outcome'
           WHERE d.event_type='decision'
           ORDER BY cost DESC LIMIT 5"""
    ).fetchall()
    for r in rows:
        preview = (r["preview"] or "")[:50]
        cost = r["cost"] or 0.0
        print(f"   ${cost:.6f}  {r['tier']:6}  {preview}")

    print()
    print("=" * 72)
    print(f" Done. Telemetry persisted to: {db_path.resolve()}")
    print(f" Open with: sqlite3 {db_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
