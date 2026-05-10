"""dmr — command-line interface for dynamic-model-router.

Subcommands:
    dmr classify "task text" [--provider google] [--config dmr.yaml]
    dmr train --data my_data.jsonl [--output model.joblib]
    dmr generate-data --domain healthcare --per-slot 30
    dmr stats [--since 24h]
    dmr init                 # scaffolds dmr.yaml in cwd
    dmr presets              # list available domain presets

Run `dmr <subcommand> --help` for details on each.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_classify(args) -> int:
    from classifier import Router

    if args.config:
        router = Router.from_yaml(args.config)
    elif args.preset:
        router = Router.from_preset(args.preset)
    else:
        router = Router()
    decision = router.classify(args.task, provider=args.provider)
    out = {
        "task": args.task[:80] + ("…" if len(args.task) > 80 else ""),
        "tier": decision.tier.value,
        "model": decision.model_name,
        "task_type": decision.task_type.value,
        "complexity": decision.complexity.value,
        "confidence": round(decision.confidence, 3),
        "layer_used": decision.layer_used,
        "compliance_flag": decision.compliance_flag,
        "reasoning": decision.reasoning,
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_train(args) -> int:
    from classifier.ml.train import train_from_data

    metadata = train_from_data(
        data_path=Path(args.data),
        output_path=Path(args.output) if args.output else None,
        max_iter=args.max_iter,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _cmd_generate_data(args) -> int:
    from classifier.ml.generate_synthetic import main as gen_main

    sys.argv = [
        "generate_synthetic",
        "--per-slot",
        str(args.per_slot),
        "--domain",
        args.domain or "",
        "--model",
        args.model,
    ]
    if args.out:
        sys.argv.extend(["--out", args.out])
    gen_main()
    return 0


def _cmd_stats(args) -> int:
    """Reuse the existing stats CLI."""
    from classifier.stats import cmd_cost, cmd_disagreements, cmd_summary

    sub = args.sub or "summary"
    handler = {
        "summary": cmd_summary,
        "disagreements": cmd_disagreements,
        "cost": cmd_cost,
    }[sub]
    handler(args)
    return 0


def _cmd_init(args) -> int:
    """Scaffold a dmr.yaml in the current working directory."""
    target = Path("dmr.yaml")
    if target.exists() and not args.force:
        print(f"Refusing to overwrite existing {target}. Use --force to replace.", file=sys.stderr)
        return 1

    template = """# yaml-language-server: $schema=https://raw.githubusercontent.com/manthan9891994/dynamic-model-router/master/classifier/dmr.schema.json

# dmr.yaml — Dynamic Model Router configuration
# Load with: Router.from_yaml("dmr.yaml")

# Provider failover order (first one tried first)
providers:
  - google
  # - anthropic
  # - openai

# Layer toggles
layer1_enabled: true
layer2_enabled: true
layer3_enabled: true

# Confidence thresholds (escalation triggers)
layer1_threshold: 0.75
layer3_threshold: 0.75

# Optional budget cap (USD/month)
# budget_usd: 1000.0

# Custom keyword packs — add domain vocabulary
# keyword_packs:
#   - name: my_domain
#     packs:
#       reasoning:
#         - my_special_term
#         - another_term
#       doc_creation:
#         - report
#         - memo
"""
    target.write_text(template, encoding="utf-8")
    print(f"Wrote {target}. Edit to customize, then load with Router.from_yaml('{target}').")
    return 0


def _cmd_eval(args) -> int:
    """Evaluate routing accuracy on a labeled JSONL file.

    Each line must have: {"task": "...", "tier": "low|medium|high"}
    Optional fields: "task_type", "complexity"

    Prints per-tier accuracy, overall accuracy, tier distribution table,
    and (optionally) per-example mismatches.
    """
    import json
    from pathlib import Path

    from classifier import Router

    path = Path(args.data)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        return 1

    router = Router.from_yaml(args.config) if args.config else Router()

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = correct = 0
    per_tier: dict[str, dict] = {}
    mismatches = []

    for raw in lines:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        task = row.get("task", "")
        expected_tier = (row.get("tier") or row.get("expected_tier") or "").lower()
        if not task or not expected_tier:
            continue

        try:
            decision = router.classify(task)
            predicted = decision.tier.value
        except Exception as exc:
            predicted = "error"
            _ = exc

        total += 1
        stats = per_tier.setdefault(expected_tier, {"total": 0, "correct": 0})
        stats["total"] += 1
        if predicted == expected_tier:
            correct += 1
            stats["correct"] += 1
        else:
            mismatches.append(
                {
                    "task": task[:80],
                    "expected": expected_tier,
                    "predicted": predicted,
                }
            )

    if total == 0:
        print('No valid rows found. Each line needs {"task": "...", "tier": "low|medium|high"}')
        return 1

    accuracy = correct / total
    print(f"\nEval results — {path.name}")
    print(f"  Overall accuracy: {accuracy:.1%}  ({correct}/{total})")
    print()
    print(f"  {'Tier':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 40}")
    for tier, s in sorted(per_tier.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {tier:<10} {s['correct']:>8} {s['total']:>8} {acc:>9.1%}")

    if mismatches and args.show_errors:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for m in mismatches[: args.limit]:
            print(f"    expected={m['expected']:6}  predicted={m['predicted']:6}  task={m['task']!r}")

    out = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_tier": {k: {"accuracy": round(v["correct"] / v["total"], 4), **v} for k, v in per_tier.items()},
    }
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n  Results saved to {args.output}")

    return 0


def _cmd_models(args) -> int:
    """Manage the model/cost/capability registry.

    Subcommands:
      list   — show currently registered providers, models, costs, capabilities
      export — write the active registry to a YAML file
      load   — load a registry from a path / URL into the runtime
      pull   — alias for `load <URL>` with progress output
      clear  — wipe all registered providers and models (start empty)
    """
    sub = (args.action or "list").lower()

    if sub == "list":
        from classifier.core.registry import MODEL_CAPABILITIES, MODEL_REGISTRY
        from classifier.infra.cost_tracker import COST_TABLE

        if not MODEL_REGISTRY:
            print("No providers registered. Try: dmr models load default")
            return 0
        print(f"Providers ({len(MODEL_REGISTRY)}):")
        for prov, tier_map in sorted(MODEL_REGISTRY.items()):
            print(f"  {prov}:")
            for tier, model in tier_map.items():
                t = tier.value if hasattr(tier, "value") else str(tier)
                cost = COST_TABLE.get(model, {})
                caps = MODEL_CAPABILITIES.get(model, {})
                cw = caps.get("context_window", "?")
                cstr = f"${cost.get('input', '?')}/${cost.get('output', '?')} per 1M" if cost else "(no cost)"
                print(f"    {t:8} -> {model:35} | {cstr} | ctx={cw}")
        return 0

    if sub == "export":
        from classifier.core.registry_loader import export_to_yaml

        out = args.output or "models.yaml"
        export_to_yaml(out)
        print(f"Wrote runtime registry to {out}.")
        return 0

    if sub in ("load", "pull"):
        from classifier.core.registry_loader import clear_registry, load_registry

        if args.replace:
            clear_registry()
        source = args.source or "default"
        meta = load_registry(source)
        print(f"Loaded {meta['providers']} providers, {meta['models']} models (version={meta['version']}).")
        return 0

    if sub == "clear":
        from classifier.core.registry_loader import clear_registry

        clear_registry()
        print("Registry cleared.")
        return 0

    print(f"Unknown action: {sub}. Use one of: list, export, load, pull, clear", file=sys.stderr)
    return 1


def _cmd_presets(args) -> int:
    from classifier.presets import available

    print("Available domain presets:")
    for name in available():
        print(f"  - {name}")
    print("\nLoad with:  Router.from_preset('<name>')")
    return 0


def _cmd_version(args) -> int:
    """Print package version + Python + key dep versions."""
    import platform

    from classifier import __version__

    out = {
        "dynamic_model_router": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for dep in (
        "pydantic",
        "pydantic_settings",
        "yaml",
        "sklearn",
        "sentence_transformers",
        "google.genai",
        "joblib",
    ):
        try:
            mod = __import__(dep.split(".")[0])
            for part in dep.split(".")[1:]:
                mod = getattr(mod, part)
            out[dep] = getattr(mod, "__version__", "installed")
        except Exception:
            out[dep] = "not installed"
    print(json.dumps(out, indent=2))
    return 0


def _cmd_relabel(args) -> int:
    """Run weak-supervision over the decision ⨝ outcome stream and emit labeled JSONL.

    Args:
        --since      ISO timestamp or relative window like "30d", "7d", "24h".
        --until      ISO timestamp upper bound (default: now).
        --min-confidence  Drop labels whose aggregated confidence is below this.
        --include-cached  Don't skip cache-hit rows (default: skip).
        --include-exploration  Don't skip exploration rows (default: skip).
        --out        Output JSONL path (default: labeled_from_telemetry.jsonl).
    """
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from classifier.ml.auto_labeler import AutoLabeler

    # Resolve --since to ISO if it's a relative window
    since = args.since
    if since:
        since = since.strip().lower()
        units = {"h": "hours", "d": "days", "w": "weeks"}
        if since[-1] in units and since[:-1].isdigit():
            now = datetime.now(timezone.utc)
            kwargs = {units[since[-1]]: int(since[:-1])}
            since = (now - timedelta(**kwargs)).isoformat()

    labeler = AutoLabeler(
        min_confidence=float(args.min_confidence),
        skip_cached=not args.include_cached,
        skip_exploration=not args.include_exploration,
    )
    rows = labeler.run(since=since, until=args.until)

    out_path = Path(args.out or "labeled_from_telemetry.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} labeled rows to {out_path}")
    print()
    print("AutoLabeler stats:")
    for k, v in sorted(labeler.stats.items()):
        print(f"  {k:24} {v}")
    return 0


def _cmd_prune(args) -> int:
    """Delete outcome rows older than --days from routing_outcomes.jsonl."""
    from classifier.infra.outcome_logger import prune_old_outcomes

    days = int(args.days)
    pruned = prune_old_outcomes(days=days)
    print(f"Pruned {pruned} outcome row(s) older than {days} days.")
    return 0


def _cmd_doctor(args) -> int:
    """Diagnose configuration issues. Reports OK/WARN/FAIL for each check."""
    import importlib

    checks: list[tuple[str, str, str]] = []  # (name, status, detail)

    def add(name, ok_or_warn, detail):
        checks.append((name, ok_or_warn, detail))

    # 1. Python version
    import sys

    pv = sys.version_info
    add(
        "Python version",
        "OK" if pv >= (3, 10) else "FAIL",
        f"{pv.major}.{pv.minor}.{pv.micro} (need >= 3.10)",
    )

    # 2. Required deps
    for mod, extra in (("pydantic_settings", "core"), ("yaml", "core"), ("dotenv", "core")):
        try:
            importlib.import_module(mod)
            add(f"dep:{mod}", "OK", "installed")
        except ImportError:
            add(f"dep:{mod}", "FAIL", f"missing — pip install dynamic-model-router[{extra}]")

    # 3. Optional deps for L2/L3
    optional = [
        ("google.genai", "google", "Layer 2 fallback"),
        ("sentence_transformers", "ml", "Layer 3 ML head"),
        ("sklearn", "ml", "Layer 3 ML head"),
        ("joblib", "ml", "Layer 3 model loading"),
    ]
    for mod, extra, purpose in optional:
        try:
            importlib.import_module(
                mod.replace(".", "_") if mod == "google.genai" else mod
            ) if mod != "google.genai" else __import__(mod)
            add(f"opt:{mod}", "OK", f"installed ({purpose})")
        except ImportError:
            add(f"opt:{mod}", "WARN", f"missing — pip install dynamic-model-router[{extra}] for {purpose}")

    # 4. .env / API keys
    try:
        from classifier.infra.config import settings

        for prov in ("google", "anthropic", "openai"):
            try:
                settings.api_key_for(prov)
                add(f"key:{prov}", "OK", "configured")
            except Exception as exc:
                add(f"key:{prov}", "WARN", str(exc).split("\n", 1)[0][:80])
        add("default_provider", "OK", settings.default_provider)
    except Exception as exc:
        add("settings", "FAIL", str(exc)[:100])

    # 5. L3 model file
    try:
        from classifier.layers.layer3 import embed_classifier

        if embed_classifier._MODEL_PATH.exists():
            sz = embed_classifier._MODEL_PATH.stat().st_size / 1024
            add("L3 model file", "OK", f"{embed_classifier._MODEL_PATH.name} ({sz:.0f}KB)")
        else:
            add("L3 model file", "WARN", "missing — run `dmr train --data ...` (L3 will abstain)")
    except Exception as exc:
        add("L3 model file", "WARN", str(exc)[:100])

    # 6. Smoke test classify
    try:
        from classifier import Router

        d = Router(layer2_enabled=False, layer3_enabled=False).classify("hello")
        add("classify smoke test", "OK", f"tier={d.tier.value} model={d.model_name}")
    except Exception as exc:
        add("classify smoke test", "FAIL", str(exc)[:100])

    # Print results
    # ASCII-only symbols (Windows cp1252 console can't encode unicode check marks)
    icon = {"OK": "+", "WARN": "!", "FAIL": "x"}
    width = max(len(n) for n, _, _ in checks) + 2
    print()
    fails = warns = 0
    for name, status, detail in checks:
        if status == "FAIL":
            fails += 1
        elif status == "WARN":
            warns += 1
        print(f"  [{icon[status]}] {name:<{width}} {status:<5} {detail}")
    print()
    print(f"  Result: {len(checks) - fails - warns} ok, {warns} warning(s), {fails} failure(s)")
    return 0 if fails == 0 else 1


def _cmd_benchmark(args) -> int:
    """Measure routing latency on synthetic input. Reports p50/p95/p99 per layer."""
    import statistics
    import time as _t

    from classifier import Router

    router = Router()
    sample_tasks = [
        "Hello, how are you?",
        "Write a Python function to merge sorted lists.",
        "Translate this to French.",
        "Calculate compound interest at 5% over 10 years.",
        "Design a CQRS architecture for healthcare records handling 10M patients with multi-region replication.",
    ]

    # Warm up
    for t in sample_tasks:
        router.classify(t)

    latencies: list[float] = []
    print(f"\n  Running {args.iterations} iterations × {len(sample_tasks)} tasks each...\n")
    for _ in range(args.iterations):
        for t in sample_tasks:
            start = _t.perf_counter()
            router.classify(t)
            latencies.append((_t.perf_counter() - start) * 1000)

    latencies.sort()
    n = len(latencies)
    print(f"  Total samples:   {n}")
    print(f"  Mean latency:    {statistics.mean(latencies):>7.2f} ms")
    print(f"  p50 (median):    {latencies[n // 2]:>7.2f} ms")
    print(f"  p95:             {latencies[int(n * 0.95)]:>7.2f} ms")
    print(f"  p99:             {latencies[int(n * 0.99)]:>7.2f} ms")
    print(f"  max:             {max(latencies):>7.2f} ms")
    print(f"  min:             {min(latencies):>7.2f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dmr",
        description="Dynamic Model Router — classify and route tasks to the right LLM tier.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # classify
    p = sub.add_parser("classify", help="Classify a single task")
    p.add_argument("task", help="The task text (use quotes)")
    p.add_argument("--provider", default=None, help="google | anthropic | openai")
    p.add_argument("--config", default=None, help="Path to dmr.yaml")
    p.add_argument("--preset", default=None, help="Preset name (healthcare, legal, fintech)")
    p.set_defaults(func=_cmd_classify)

    # train
    p = sub.add_parser("train", help="Train Stage 2 head on a JSONL dataset")
    p.add_argument("--data", required=True, help="Path to JSONL training data")
    p.add_argument("--output", default=None, help="Where to save model bundle")
    p.add_argument("--max-iter", type=int, default=600, help="MLP max iterations")
    p.set_defaults(func=_cmd_train)

    # generate-data
    p = sub.add_parser("generate-data", help="Generate synthetic training data via Gemini")
    p.add_argument("--per-slot", type=int, default=30)
    p.add_argument("--domain", default="", help="Optional: healthcare | fintech | legal")
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_generate_data)

    # stats
    p = sub.add_parser("stats", help="Routing statistics from decision log")
    p.add_argument(
        "sub",
        nargs="?",
        default="summary",
        choices=["summary", "disagreements", "cost"],
        help="Which view (default: summary)",
    )
    p.add_argument("--since", default="24h", help="Window: 60m | 24h | 7d | 30d")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cmd_stats)

    # init
    p = sub.add_parser("init", help="Scaffold a dmr.yaml in current directory")
    p.add_argument("--force", action="store_true", help="Overwrite existing file")
    p.set_defaults(func=_cmd_init)

    # eval
    p = sub.add_parser("eval", help="Evaluate routing accuracy on a labeled JSONL file")
    p.add_argument("--data", required=True, help='JSONL file with {"task": "...", "tier": "low|medium|high"}')
    p.add_argument("--config", default=None, help="Path to dmr.yaml")
    p.add_argument("--output", default=None, help="Save JSON results to this file")
    p.add_argument("--show-errors", action="store_true", help="Print mismatched examples")
    p.add_argument("--limit", type=int, default=20, help="Max mismatches to print (default 20)")
    p.set_defaults(func=_cmd_eval)

    # presets
    p = sub.add_parser("presets", help="List available domain presets")
    p.set_defaults(func=_cmd_presets)

    # models — registry management
    p = sub.add_parser("models", help="Manage the model/cost/capability registry")
    p.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "export", "load", "pull", "clear"],
        help="What to do (default: list)",
    )
    p.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Registry source for load/pull (path, URL, or 'default'/'empty')",
    )
    p.add_argument("--output", default=None, help="Output path for export (default: models.yaml)")
    p.add_argument(
        "--replace", action="store_true", help="Clear runtime registry before loading (instead of merging)"
    )
    p.set_defaults(func=_cmd_models)

    # version
    p = sub.add_parser("version", help="Print package version + dependencies")
    p.set_defaults(func=_cmd_version)

    # doctor
    p = sub.add_parser("doctor", help="Diagnose configuration and dependency issues")
    p.set_defaults(func=_cmd_doctor)

    # prune — outcome log retention
    p = sub.add_parser("prune", help="Delete outcome rows older than N days")
    p.add_argument("--days", type=int, default=90, help="Retention window in days (default: 90)")
    p.set_defaults(func=_cmd_prune)

    # relabel — auto-label decision ⨝ outcome stream → training JSONL
    p = sub.add_parser("relabel", help="Weak-supervised auto-label production telemetry")
    p.add_argument("--since", default=None, help="Relative window (e.g. 30d, 7d, 24h) or ISO timestamp")
    p.add_argument("--until", default=None, help="ISO upper bound (default: now)")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Drop labels below this aggregated confidence (default 0.7)",
    )
    p.add_argument("--include-cached", action="store_true", help="Include cache-hit rows (default: skipped)")
    p.add_argument(
        "--include-exploration", action="store_true", help="Include exploration rows (default: skipped)"
    )
    p.add_argument("--out", default=None, help="Output JSONL path (default: labeled_from_telemetry.jsonl)")
    p.set_defaults(func=_cmd_relabel)

    # benchmark
    p = sub.add_parser("benchmark", help="Measure routing latency p50/p95/p99")
    p.add_argument("--iterations", type=int, default=20, help="iterations per sample task (default 20)")
    p.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
