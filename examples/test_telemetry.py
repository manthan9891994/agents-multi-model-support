"""Practical end-to-end test of the telemetry system.

Run this file 3 ways to see each mode in action:

    # 1. Default — quiet INFO lines, no files, no DB
    python examples/test_telemetry.py

    # 2. Full telemetry — DEBUG JSON via Python logging
    DMR_TELEMETRY=1 python examples/test_telemetry.py

    # 3. Custom DB backend — SQLite (see examples/custom_backends/sqlite_backend.py)
    python examples/test_telemetry.py --db

PowerShell equivalent for #2:
    $env:DMR_TELEMETRY="1"; python examples/test_telemetry.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Make the local repo win over any installed `classifier` package
sys.path.insert(0, str(Path(__file__).parent.parent))
# Make the SQLite example importable
sys.path.insert(0, str(Path(__file__).parent))

from classifier import MultiLoggerBackend, OutcomeRecord, Router, StdoutLoggerBackend
from classifier.infra.outcome_logger import log_outcome


# A few realistic tasks across tiers
TASKS = [
    "explain recursion in Python",
    "translate this paragraph to French",
    "compare the carbon footprint of solar vs nuclear and identify regulatory risks",
    "what is the capital of Japan",
    "design a fault-tolerant distributed cache with strong consistency guarantees",
]


def setup_logging() -> None:
    """Show INFO+ on stdout — this is what a user app would do."""
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("DMR_TELEMETRY") else logging.INFO,
        format="%(levelname)-5s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # silence noisy 3rd-party deps
    for n in ("urllib3", "httpcore", "httpx", "filelock", "sentence_transformers", "asyncio"):
        logging.getLogger(n).setLevel(logging.WARNING)


def run_router(router: Router) -> list:
    """Classify all tasks, simulate outcomes, return decisions."""
    decisions = []
    for task in TASKS:
        d = router.classify(task)
        decisions.append(d)
        # Fake an outcome — in real usage your LLM call would report tokens
        log_outcome(
            OutcomeRecord(
                decision_id=d.decision_id,
                tokens_in=len(task.split()) * 2,
                tokens_out=80 if d.tier.value == "low" else 250,
                wall_ms=300.0 if d.tier.value == "low" else 800.0,
                success=True,
                cost_usd=0.00001 if d.tier.value == "low" else 0.00015,
            )
        )
    return decisions


def mode_default() -> None:
    print("\n" + "=" * 70)
    print("MODE 1: Default — no DMR_TELEMETRY, no backend")
    print("=" * 70)
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    run_router(router)


def mode_telemetry_env() -> None:
    print("\n" + "=" * 70)
    print(f"MODE 2: DMR_TELEMETRY={os.getenv('DMR_TELEMETRY')} — full structured DEBUG")
    print("=" * 70)
    router = Router(layer2_enabled=False, layer3_enabled=False, cache_enabled=False)
    run_router(router)


def mode_custom_db() -> None:
    """Plug in the SQLite example backend, then query it back."""
    from custom_backends.sqlite_backend import SQLiteBackend

    db_path = Path("_demo_telemetry.db")
    if db_path.exists():
        db_path.unlink()

    print("\n" + "=" * 70)
    print("MODE 3: Custom SQLite backend (user-managed storage)")
    print("=" * 70)

    sqlite_backend = SQLiteBackend(str(db_path))

    # Fan out: stdout (for visibility) + SQLite (for persistence)
    multi = MultiLoggerBackend([sqlite_backend, StdoutLoggerBackend()])

    router = Router(
        decision_logger=multi,
        outcome_logger=multi,
        layer2_enabled=False,
        layer3_enabled=False,
        cache_enabled=False,
    )

    print("\n--- Routing 5 tasks (fan-out: SQLite + stdout) ---\n")
    decisions = run_router(router)

    # Now query what we stored
    print("\n--- Reading back from SQLite ---")
    stored_decisions = sqlite_backend.read()
    print(f"Stored {len(stored_decisions)} decisions in {db_path}")

    decision_ids = {d.decision_id for d in decisions}
    stored_outcomes = sqlite_backend.read(decision_ids=decision_ids)
    print(f"Stored {len(stored_outcomes)} outcomes\n")

    # Quick analytics — what you'd put in a real dashboard
    print("--- Analytics ---")
    tier_counts: dict = {}
    total_cost = 0.0
    total_tokens_in = total_tokens_out = 0
    for row in stored_decisions:
        tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
    for row in stored_outcomes:
        total_cost += row.get("cost_usd") or 0
        total_tokens_in += row.get("tokens_in") or 0
        total_tokens_out += row.get("tokens_out") or 0

    print(f"Tier distribution:  {tier_counts}")
    print(f"Total tokens in:    {total_tokens_in}")
    print(f"Total tokens out:   {total_tokens_out}")
    print(f"Total cost (USD):   ${total_cost:.6f}")

    # Show a joined row
    print("\n--- Sample joined row (decision + outcome) ---")
    sample_id = decisions[0].decision_id
    decision = next(d for d in stored_decisions if d["decision_id"] == sample_id)
    outcome = next(o for o in stored_outcomes if o["decision_id"] == sample_id)
    print(json.dumps({"decision": decision, "outcome": outcome}, indent=2))


if __name__ == "__main__":
    setup_logging()

    use_db = "--db" in sys.argv
    has_env = bool(os.getenv("DMR_TELEMETRY"))

    if use_db:
        mode_custom_db()
    elif has_env:
        mode_telemetry_env()
    else:
        mode_default()

    print("\nDone. Try the other modes:")
    print("  DMR_TELEMETRY=1 python examples/test_telemetry.py   # full JSON")
    print("  python examples/test_telemetry.py --db              # SQLite backend")
