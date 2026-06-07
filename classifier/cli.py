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

    if args.auto:
        return _cmd_train_auto(args)

    if not args.data:
        print(
            "Error: provide --data <path.jsonl> OR --auto (bootstrap from logs).",
            file=sys.stderr,
        )
        return 2

    metadata = train_from_data(
        data_path=Path(args.data),
        output_path=Path(args.output) if args.output else None,
        max_iter=args.max_iter,
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _cmd_train_auto(args) -> int:
    """`dmr train --auto` — train Layer 3 from production telemetry.

    Pipeline:
        1. AutoLabeler reads routing_decisions.jsonl ⨝ routing_outcomes.jsonl
           and applies 8 weak-supervision rules (Snorkel-style) to produce
           weighted labels.
        2. Drops rows below --min-confidence (default 0.7).
        3. Writes a temp JSONL and feeds it to the standard train pipeline.
        4. Runs `dmr eval` on a held-out slice and prints the headline number.

    Zero-config: just `dmr train --auto`. Run it again whenever you have more
    data — each run replaces the model.
    """
    import tempfile
    from datetime import datetime, timedelta, timezone

    from classifier.ml.auto_labeler import AutoLabeler
    from classifier.ml.embeddings import ensure_encoder_available
    from classifier.ml.train import train_from_data

    # Resolve the encoder up front. With the bundled default this returns
    # instantly; only a custom DMR_EMBEDDING_MODEL would trigger a download.
    print("[0/3] Resolving encoder (bundled by default — no network needed)...")
    if ensure_encoder_available() is None:
        print(
            "  Encoder unavailable — install ML extras with "
            "`pip install dynamic-model-router[ml]` and retry.",
            file=sys.stderr,
        )
        return 2

    # Resolve --since to ISO if it's a relative window (default: last 90 days)
    since = (args.since or "90d").strip().lower()
    units = {"h": "hours", "d": "days", "w": "weeks"}
    if since[-1] in units and since[:-1].isdigit():
        now = datetime.now(timezone.utc)
        kw = {units[since[-1]]: int(since[:-1])}
        since = (now - timedelta(**kw)).isoformat()

    print(f"[1/3] Auto-labeling decision/outcome telemetry since {since[:10]}...")
    labeler = AutoLabeler(min_confidence=float(args.min_confidence))
    rows = labeler.run(since=since)

    if len(rows) < 50:
        print(
            f"\n  Only {len(rows)} confident labels found. Need >= 50.\n"
            f"  Keep using the router and re-run after more decisions accumulate.\n"
            f"  Tip: lower --min-confidence (default 0.7) to harvest more rows,\n"
            f"  or run `dmr generate-data` to bootstrap with synthetic examples.",
            file=sys.stderr,
        )
        return 1

    # Show class distribution so users know what's in their data
    print(f"  Got {len(rows)} confident labels:")
    from collections import Counter

    tt_counts = Counter(r.get("task_type") for r in rows if r.get("task_type"))
    cx_counts = Counter(r.get("complexity") for r in rows if r.get("complexity"))
    for tt, n in tt_counts.most_common():
        print(f"    task_type   {tt:<20} {n}")
    for cx, n in cx_counts.most_common():
        print(f"    complexity  {cx:<20} {n}")

    print("\n[2/3] Training Layer 3 head (frozen MiniLM + calibrated MLPs)...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        for r in rows:
            tmp.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path = Path(tmp.name)

    try:
        metadata = train_from_data(
            data_path=tmp_path,
            output_path=Path(args.output) if args.output else None,
            max_iter=args.max_iter,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata["bootstrap_source"] = "auto-from-telemetry"
    metadata["bootstrap_window"] = args.since or "90d"
    metadata["bootstrap_min_confidence"] = float(args.min_confidence)

    print("\n[3/3] Done.")
    print(json.dumps(metadata, indent=2))
    print(
        "\n  Layer 3 is now active. New `Router()` instances will pick it up\n"
        "  automatically when constructed with `layer3_enabled='auto'` (default)."
    )
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

    # 5. L3 model file + data-readiness suggestion
    try:
        from classifier.layers.layer3 import embed_classifier

        if embed_classifier._MODEL_PATH.exists():
            sz = embed_classifier._MODEL_PATH.stat().st_size / 1024
            add("L3 model file", "OK", f"{embed_classifier._MODEL_PATH.name} ({sz:.0f}KB)")
        else:
            # Count decisions logged so far — suggest training when enough exist
            try:
                from classifier.infra.decision_logger import read_decisions

                n_decisions = len(read_decisions())
            except Exception:
                n_decisions = 0
            if n_decisions >= 200:
                add(
                    "L3 model file",
                    "WARN",
                    f"missing, but {n_decisions} decisions logged — run `dmr train --auto` to enable L3",
                )
            elif n_decisions >= 50:
                add(
                    "L3 model file",
                    "WARN",
                    f"missing — {n_decisions} decisions logged so far (need ~200 for `dmr train --auto`)",
                )
            else:
                add(
                    "L3 model file",
                    "WARN",
                    f"missing — keep using the router; train later with `dmr train --auto` (have {n_decisions} decisions)",
                )
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


def _user_keywords_dir() -> Path:
    """User's persistent keyword pack directory (~/.dmr/keywords).

    Honors DMR_KEYWORDS_DIR env var so tests can isolate state.
    """
    import os

    env = os.environ.get("DMR_KEYWORDS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".dmr" / "keywords"


def _load_user_keyword_pack(domain: str) -> dict:
    """Read ~/.dmr/keywords/<domain>.yaml or return empty skeleton."""
    import yaml

    path = _user_keywords_dir() / f"{domain}.yaml"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("name", domain)
    data.setdefault("task_keywords", {})
    data.setdefault("escalators", {})
    return data


def _save_user_keyword_pack(domain: str, data: dict) -> Path:
    import yaml

    d = _user_keywords_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{domain}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


def _cmd_keywords(args) -> int:
    """`dmr keywords` — easy keyword authoring.

    Subcommands:
        dmr keywords add --domain legal --type reasoning --keywords "tort,liable"
        dmr keywords list [--domain legal]
        dmr keywords remove --domain legal --keyword "tort"
        dmr keywords suggest [--since 7d] [--top 20]

    Packs persist at ~/.dmr/keywords/<domain>.yaml and are auto-loaded on
    Router() construction (no code change needed).
    """
    if args.action == "list":
        return _cmd_keywords_list(args)
    if args.action == "add":
        return _cmd_keywords_add(args)
    if args.action == "remove":
        return _cmd_keywords_remove(args)
    if args.action == "suggest":
        return _cmd_keywords_suggest(args)
    print(f"Unknown action: {args.action}", file=sys.stderr)
    return 2


def _cmd_keywords_list(args) -> int:
    d = _user_keywords_dir()
    files = sorted(d.glob("*.yaml")) if d.exists() else []
    if not files and not args.domain:
        print(
            "No user keyword packs yet. Add some with:\n"
            '  dmr keywords add --domain <name> --type <task_type> --keywords "a,b,c"'
        )
        return 0
    if args.domain:
        files = [f for f in files if f.stem == args.domain]
        if not files:
            print(f"No pack named '{args.domain}' at {d / (args.domain + '.yaml')}")
            return 1
    for f in files:
        data = _load_user_keyword_pack(f.stem)
        print(f"\n[{f.stem}]   ({f})")
        for tt, groups in (data.get("task_keywords") or {}).items():
            for grp, kws in (groups or {}).items():
                print(f"  {tt:<18} {grp:<10} {', '.join(kws)}")
        for kw, w in (data.get("escalators") or {}).items():
            print(f"  escalator {' ':<8} weight={w:<3} {kw}")
    return 0


def _cmd_keywords_add(args) -> int:
    if not args.domain or not args.type or not args.keywords:
        print(
            "Error: --domain, --type, and --keywords are required.\n"
            'Example: dmr keywords add --domain legal --type reasoning --keywords "tort,liable"',
            file=sys.stderr,
        )
        return 2

    new_kws = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    if not new_kws:
        print("Error: --keywords was empty after parsing.", file=sys.stderr)
        return 2

    # Validate task_type
    from classifier.core.types import TaskType, task_type_for

    try:
        task_type_for(args.type)
    except (KeyError, ValueError):
        valid = sorted(t.value for t in TaskType)
        print(
            f"Error: '{args.type}' is not a known task type.\n  Valid: {', '.join(valid)}",
            file=sys.stderr,
        )
        return 2

    data = _load_user_keyword_pack(args.domain)
    slot = data["task_keywords"].setdefault(args.type, {})
    existing = slot.setdefault(args.group, [])

    # Conflict check — flag if a keyword already lives in another task_type
    conflicts: list[str] = []
    for tt, groups in data["task_keywords"].items():
        if tt == args.type:
            continue
        for _grp, kws in (groups or {}).items():
            for kw in new_kws:
                if kw in (kws or []):
                    conflicts.append(f"  '{kw}' already in {tt}")

    added = 0
    for kw in new_kws:
        if kw not in existing:
            existing.append(kw)
            added += 1

    path = _save_user_keyword_pack(args.domain, data)
    print(f"  + Added {added} keyword(s) to [{args.domain}] / {args.type} / {args.group}")
    print(f"    -> {path}")
    if conflicts:
        print("\n  ! Conflict warnings (same keyword in another task_type):")
        for c in conflicts:
            print(c)
    print("\n  These will be active on the next Router() construction. No code change needed.")
    return 0


def _cmd_keywords_remove(args) -> int:
    if not args.domain or not args.keyword:
        print(
            "Error: --domain and --keyword are required.\n"
            'Example: dmr keywords remove --domain legal --keyword "tort"',
            file=sys.stderr,
        )
        return 2
    data = _load_user_keyword_pack(args.domain)
    target = args.keyword.strip().lower()
    removed = 0
    for tt, groups in list(data.get("task_keywords", {}).items()):
        for grp, kws in list((groups or {}).items()):
            if target in (kws or []):
                kws.remove(target)
                removed += 1
                print(f"  - Removed '{target}' from {tt}/{grp}")
    if data.get("escalators", {}).pop(target, None) is not None:
        removed += 1
        print(f"  - Removed escalator '{target}'")
    if removed == 0:
        print(f"  '{target}' not found in [{args.domain}].")
        return 1
    _save_user_keyword_pack(args.domain, data)
    return 0


def _cmd_keywords_suggest(args) -> int:
    """Mine n-grams from routing_decisions.jsonl that strongly correlate with each task_type."""
    from classifier.ml.keyword_miner import suggest_keywords

    suggestions = suggest_keywords(
        since=args.since,
        top_per_type=int(args.top),
        min_occurrences=int(args.min_occurrences),
    )

    if not suggestions:
        print(
            "No suggestions yet. Need more decisions in routing_decisions.jsonl\n"
            "(at least ~50 per task_type for meaningful TF-IDF)."
        )
        return 0

    print("Top distinctive n-grams per task_type (not already in any pack):\n")
    for tt, items in suggestions.items():
        if not items:
            continue
        print(f"  [{tt}]")
        for kw, score, count in items:
            print(f"    {score:5.2f}   n={count:<4}   {kw}")
        print()
    print(
        "  Tip: pick the strongest ones and add with\n"
        '    dmr keywords add --domain <name> --type <task_type> --keywords "kw1,kw2"'
    )
    return 0


def _cmd_config(args) -> int:
    """`dmr config show|validate` — easy inspection of the running config."""
    if args.action == "show":
        return _cmd_config_show(args)
    if args.action == "validate":
        return _cmd_config_validate(args)
    return 2


def _cmd_config_show(args) -> int:
    """Print the effective config: settings, registry, packs, model file status."""
    from classifier import __version__
    from classifier.core import registry as _reg
    from classifier.infra.config import settings
    from classifier.layers.layer1.pack_registry import list_registered as _list_registered
    from classifier.router import _l3_model_available

    print(f"\n  dynamic-model-router  v{__version__}\n")
    print("  [settings]")
    print(f"    default_provider          {settings.default_provider}")
    print(f"    layer1_enabled            {getattr(settings, 'layer1_enabled', True)}")
    print(f"    layer2_enabled            {settings.layer2_enabled}")
    print(f"    layer3_enabled            {settings.layer3_enabled}")
    print(f"    layer2_confidence_thresh  {settings.layer2_confidence_threshold}")
    print(f"    layer3_confidence_thresh  {settings.layer3_confidence_threshold}")
    print(f"    cache_enabled             {settings.cache_enabled}")
    print(f"    monthly_budget_usd        ${settings.monthly_budget_usd}")
    print()
    print("  [registry]")
    print(f"    providers                 {', '.join(_reg.list_providers()) or '(none)'}")
    print(f"    models                    {len(_reg.list_models())}")
    print()
    print("  [layer 3]")
    if _l3_model_available():
        from classifier.layers.layer3 import embed_classifier as _ec

        sz = _ec._MODEL_PATH.stat().st_size / 1024
        print(f"    model file                {_ec._MODEL_PATH.name} ({sz:.0f} KB)")
        meta = _ec._MODEL_PATH.with_suffix(".metadata.json")
        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                if "n_examples" in m:
                    print(f"    trained on                {m['n_examples']} examples")
                if "task_type_test_accuracy" in m:
                    print(f"    task_type accuracy        {m['task_type_test_accuracy']:.3f}")
                if "complexity_test_accuracy" in m:
                    print(f"    complexity accuracy       {m['complexity_test_accuracy']:.3f}")
            except Exception:
                pass
    else:
        print("    model file                (not trained — run `dmr train --auto`)")
    print()
    user_packs = (
        _list_registered() + [p.stem for p in _user_keywords_dir().glob("*.yaml")]
        if _user_keywords_dir().exists()
        else _list_registered()
    )
    print("  [keyword packs]")
    print(f"    registered                {', '.join(user_packs) if user_packs else '(built-in only)'}")
    print()
    return 0


def _cmd_config_validate(args) -> int:
    """Validate dmr.yaml against the bundled JSON schema."""
    cfg = Path(args.config) if args.config else Path("dmr.yaml")
    if not cfg.exists():
        print(f"Error: {cfg} not found.", file=sys.stderr)
        return 2

    import yaml

    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"YAML syntax error: {exc}", file=sys.stderr)
        return 1

    schema_path = Path(__file__).parent / "dmr.schema.json"
    if not schema_path.exists():
        print(f"  YAML parses cleanly. (Schema not bundled at {schema_path}; skipping deep validation.)")
        return 0

    try:
        import jsonschema
    except ImportError:
        print("  YAML parses cleanly. Install `jsonschema` for full validation:\n    pip install jsonschema")
        return 0

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        print(f"  ✗ {cfg} fails schema validation:")
        print(f"    path:    /{'/'.join(str(p) for p in exc.absolute_path)}")
        print(f"    error:   {exc.message}")
        return 1
    print(f"  + {cfg} is valid against the dmr.yaml schema.")
    return 0


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


def _cmd_frontier(args) -> int:
    """Show the cost↔posture frontier: per savings_level, which levers turn on and
    the projected relative cost on a nominal multi-step (input-dominant) turn.

    Projects the model-stable cost levers (cache / context-prune / effort) which are
    quality-neutral. Quality at aggressive levels (3+) must be verified with a judge.
    """
    from classifier.infra.config import settings
    from classifier.routing.posture import apply_posture

    profile = [(2000, 80)] * 8 + [(4000, 600)]  # 8 light dispatch calls + 1 heavy synthesis
    fields = (
        "dmr_cache_aware",
        "dmr_context_reduction",
        "dmr_effort_routing",
        "dmr_model_routing",
        "dmr_escalate_on_failure",
    )
    saved = {f: getattr(settings, f) for f in fields}

    def _reset():
        settings.dmr_cache_aware = False
        settings.dmr_context_reduction = "off"
        settings.dmr_effort_routing = False
        settings.dmr_model_routing = "off"
        settings.dmr_escalate_on_failure = False

    def _project() -> float:
        c_in = 0.6 if settings.dmr_context_reduction == "prune" else 1.0
        c_out = 0.8 if settings.dmr_effort_routing else 1.0
        cached = 0.7 if settings.dmr_cache_aware else 0.0
        total = 0.0
        for i, (ti, to) in enumerate(profile):
            eff_in = ti * ((1 - cached) + cached * 0.25) if i > 0 else ti
            total += eff_in * c_in * 1.25 / 1e6 + to * c_out * 10.0 / 1e6
        return total

    names = ["Off", "Saver", "Balanced", "Aggressive", "Max"]
    base = None
    print(f"\n  {'lvl':<4}{'name':<12}{'levers':<46}{'rel cost':>9}")
    print(f"  {'-' * 70}")
    for lvl in range(5):
        _reset()
        apply_posture(lvl)
        cost = _project()
        base = base or cost
        levers = []
        if settings.dmr_cache_aware:
            levers.append("cache")
        if settings.dmr_effort_routing:
            levers.append("effort")
        if settings.dmr_context_reduction != "off":
            levers.append("prune")
        if settings.dmr_model_routing != "off":
            levers.append("dispatch-downgrade")
        if settings.dmr_escalate_on_failure:
            levers.append("escalate")
        print(f"  {lvl:<4}{names[lvl]:<12}{(', '.join(levers) or '-'):<46}{cost / base * 100:>8.0f}%")
    for f, v in saved.items():
        setattr(settings, f, v)
    print(
        "\n  Projection of model-stable cost levers on an input-dominant turn.\n"
        "  Levels 1-2 are quality-neutral; verify levels 3+ with a judge on your workload.\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dmr",
        description="Dynamic Model Router — classify and route tasks to the right LLM tier.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # frontier
    p = sub.add_parser("frontier", help="Show the cost↔savings posture frontier (levels 0–4)")
    p.set_defaults(func=_cmd_frontier)

    # classify
    p = sub.add_parser("classify", help="Classify a single task")
    p.add_argument("task", help="The task text (use quotes)")
    p.add_argument("--provider", default=None, help="google | anthropic | openai")
    p.add_argument("--config", default=None, help="Path to dmr.yaml")
    p.add_argument("--preset", default=None, help="Preset name (healthcare, legal, fintech)")
    p.set_defaults(func=_cmd_classify)

    # train
    p = sub.add_parser(
        "train",
        help="Train the Layer 3 head from a JSONL file or auto-bootstrap from production logs",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Path to a JSONL training file (skip with --auto)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Bootstrap from routing_decisions.jsonl + routing_outcomes.jsonl using "
        "weak supervision (Snorkel-style). Zero-config training.",
    )
    p.add_argument(
        "--since",
        default=None,
        help="With --auto: window of telemetry to label (e.g. 90d, 30d, 7d). Default: 90d.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="With --auto: drop labels below this aggregated confidence (default 0.7)",
    )
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

    # keywords — author and persist L1 keyword packs
    p = sub.add_parser(
        "keywords",
        help="Add/list/remove/suggest L1 keyword packs (saved to ~/.dmr/keywords/)",
    )
    p.add_argument(
        "action",
        choices=["add", "list", "remove", "suggest"],
        help="Subcommand",
    )
    p.add_argument("--domain", default=None, help="Pack name (e.g. legal, finops)")
    p.add_argument("--type", default=None, help="task_type for `add` (e.g. reasoning)")
    p.add_argument("--group", default="primary", help='"primary" (default) or "secondary"')
    p.add_argument("--keywords", default=None, help='Comma-separated: "tort,liable,precedent"')
    p.add_argument("--keyword", default=None, help="Single keyword (for `remove`)")
    p.add_argument("--since", default="30d", help="Mining window for `suggest` (e.g. 7d, 30d)")
    p.add_argument("--top", type=int, default=15, help="`suggest`: top-N per task_type (default 15)")
    p.add_argument("--min-occurrences", type=int, default=3, help="`suggest`: drop n-grams seen fewer times")
    p.set_defaults(func=_cmd_keywords)

    # config — easy inspection / validation
    p = sub.add_parser("config", help="Inspect or validate the active configuration")
    p.add_argument("action", choices=["show", "validate"], help="Subcommand")
    p.add_argument("--config", default=None, help="Path to dmr.yaml (default: cwd)")
    p.set_defaults(func=_cmd_config)

    # benchmark
    p = sub.add_parser("benchmark", help="Measure routing latency p50/p95/p99")
    p.add_argument("--iterations", type=int, default=20, help="iterations per sample task (default 20)")
    p.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
