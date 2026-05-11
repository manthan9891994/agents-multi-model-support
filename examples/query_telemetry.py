"""Quick SQL query helper for the telemetry DB (no sqlite3 CLI needed).

Usage:
    python examples/query_telemetry.py "SELECT tier, COUNT(*) FROM events WHERE event_type='decision' GROUP BY tier"
    python examples/query_telemetry.py --db _support_app.db "SELECT * FROM events LIMIT 3"

Or use as an interactive REPL:
    python examples/query_telemetry.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys


def run_query(db_path: str, sql: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(sql)
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}", file=sys.stderr)
        sys.exit(1)
    cols = [c[0] for c in cur.description] if cur.description else []
    rows = cur.fetchall()
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(c), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(f"\n{len(rows)} row(s)")


def repl(db_path: str) -> None:
    print(f"Telemetry DB: {db_path}")
    print("Type SQL queries (one per line). Empty line to exit. .schema to view tables.\n")
    conn = sqlite3.connect(db_path)
    while True:
        try:
            sql = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not sql:
            break
        if sql == ".schema":
            for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table'"):
                print(row[0])
            continue
        if sql == ".tables":
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                print(row[0])
            continue
        try:
            run_query(db_path, sql)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="_support_app.db", help="Path to telemetry SQLite DB")
    ap.add_argument("sql", nargs="?", help="SQL query (omit to enter REPL)")
    args = ap.parse_args()
    if args.sql:
        run_query(args.db, args.sql)
    else:
        repl(args.db)
