"""Decision-log analyzer CLI.

Usage:
    python -m classifier.stats summary [--since 24h|7d|30d]
    python -m classifier.stats disagreements [--since 7d] [--limit 20]
    python -m classifier.stats cost [--since 30d]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_LOG_FILE = Path(__file__).parent.parent / "routing_decisions.jsonl"


def _parse_since(since: str) -> datetime:
    now = datetime.now(timezone.utc)
    since = since.strip()
    if since.endswith("h"):
        return now - timedelta(hours=int(since[:-1]))
    if since.endswith("d"):
        return now - timedelta(days=int(since[:-1]))
    return now - timedelta(days=7)


def _load_records(since: str) -> list[dict]:
    cutoff  = _parse_since(since)
    records = []
    if not _LOG_FILE.exists():
        print(f"Log file not found: {_LOG_FILE}", file=sys.stderr)
        return records
    with open(_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts  = datetime.fromisoformat(rec["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    records.append(rec)
            except Exception:
                continue
    return records


def cmd_summary(args) -> None:
    records = _load_records(args.since)
    if not records:
        print(f"No records since {args.since}.")
        return

    total        = len(records)
    layer_counts = Counter(r.get("layer", "layer1") for r in records)
    l1_only      = layer_counts.get("layer1", 0)
    l2_fired     = layer_counts.get("layer2", 0)
    pii_count    = sum(1 for r in records if r.get("compliance_flag"))
    disagree_ct  = sum(1 for r in records if r.get("disagreement"))

    type_counts: Counter  = Counter(r.get("task_type", "?") for r in records)
    tier_latencies: dict  = defaultdict(list)
    for r in records:
        tier_latencies[r.get("tier", "?")].append(float(r.get("latency_ms", 0)))

    print(f"\n=== Summary (since {args.since}) ===")
    print(f"Total:          {total:>8,}")
    print(f"L1-only:        {l1_only:>8,}  ({l1_only/total*100:.0f}%)")
    print(f"L2-fired:       {l2_fired:>8,}  ({l2_fired/total*100:.0f}%)")
    print(f"PII-flagged:    {pii_count:>8,}  ({pii_count/total*100:.1f}%)")
    print(f"L1≠L2 disagree: {disagree_ct:>8,}  ({disagree_ct/total*100:.1f}%)")

    print("\nAvg classifier latency by tier:")
    for tier_name in ("low", "medium", "high"):
        lats = tier_latencies.get(tier_name, [])
        if lats:
            print(f"  {tier_name.upper():8s}: {sum(lats)/len(lats):.1f}ms avg")

    print("\nTop task types:")
    for tt, cnt in type_counts.most_common(5):
        print(f"  {tt:25s}: {cnt:>6,}  ({cnt/total*100:.0f}%)")


def cmd_disagreements(args) -> None:
    records  = _load_records(args.since)
    disagree = [r for r in records if r.get("disagreement")]
    print(f"\n=== Disagreements (since {args.since}) — {len(disagree)} total, showing {args.limit} ===")
    if not disagree:
        print("  (none)")
        return
    for r in disagree[:args.limit]:
        ts   = r.get("timestamp", "?")[:19]
        prev = r.get("task_preview", "")[:80]
        print(f"  [{ts}] {prev}")
        print(
            f"    → {r.get('task_type','?')} / {r.get('complexity','?')} / "
            f"{r.get('tier','?').upper()} (conf={r.get('confidence','?')})"
        )


def cmd_cost(args) -> None:
    records = _load_records(args.since)
    try:
        from classifier.infra.cost_tracker import COST_PER_1M_TOKENS
    except ImportError:
        COST_PER_1M_TOKENS = {}

    total_cost:  float = 0.0
    model_costs: dict  = defaultdict(float)
    for r in records:
        model = r.get("model", "")
        rate  = COST_PER_1M_TOKENS.get(model, 0.25)
        cost  = (500 / 1_000_000) * rate  # approximate per-call cost
        model_costs[model] += cost
        total_cost += cost

    print(f"\n=== Estimated Routing Cost (since {args.since}) ===")
    print(f"Total (classifier overhead, approx): ${total_cost:.4f}")
    print("\nBy model:")
    for model, cost in sorted(model_costs.items(), key=lambda x: -x[1]):
        print(f"  {model:45s}: ${cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m classifier.stats")
    sub    = parser.add_subparsers(dest="cmd")

    s = sub.add_parser("summary", help="High-level routing statistics")
    s.add_argument("--since", default="24h", help="Time window: 24h, 7d, 30d")

    d = sub.add_parser("disagreements", help="Tasks where L1 and L2 disagreed")
    d.add_argument("--since", default="7d")
    d.add_argument("--limit", type=int, default=20)

    c = sub.add_parser("cost", help="Estimated classifier routing cost")
    c.add_argument("--since", default="30d")

    args = parser.parse_args()
    if args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "disagreements":
        cmd_disagreements(args)
    elif args.cmd == "cost":
        cmd_cost(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
