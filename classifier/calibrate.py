"""Offline confidence calibration script.

Reads routing_decisions.jsonl + reference_tasks.jsonl to compute per-bucket accuracy,
then writes classifier/data/calibration.json.

Usage:
    python -m classifier.calibrate
"""
import json
import math
from collections import defaultdict
from pathlib import Path

_DECISIONS_FILE  = Path(__file__).parent.parent / "routing_decisions.jsonl"
_REFERENCE_FILE  = Path(__file__).parent / "data" / "reference_tasks.jsonl"
_CALIBRATION_OUT = Path(__file__).parent / "data" / "calibration.json"


def _bucket(conf: float) -> str:
    lo = math.floor(conf * 10) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def run_calibration() -> dict:
    if not _REFERENCE_FILE.exists():
        print("No reference_tasks.jsonl found — nothing to calibrate.")
        return {}

    ground_truth: dict[str, tuple[str, str]] = {}
    with open(_REFERENCE_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = rec["task"][:100]
                ground_truth[key] = (rec["expected_type"], rec["expected_complexity"])
            except Exception:
                continue

    if not ground_truth:
        print("Reference file is empty.")
        return {}

    if not _DECISIONS_FILE.exists():
        print("No routing_decisions.jsonl found.")
        return {}

    # {layer: {bucket: [correct_count, total_count]}}
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    with open(_DECISIONS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec   = json.loads(line)
                key   = rec.get("task_preview", "")[:100]
                if key not in ground_truth:
                    continue
                exp_type, exp_cx = ground_truth[key]
                layer   = rec.get("layer", "layer1")
                conf    = float(rec.get("confidence", 0.5))
                correct = int(
                    rec.get("task_type") == exp_type
                    and rec.get("complexity") == exp_cx
                )
                buckets[layer][_bucket(conf)][0] += correct
                buckets[layer][_bucket(conf)][1] += 1
            except Exception:
                continue

    calibration: dict = {}
    for layer, bmap in buckets.items():
        calibration[layer] = {}
        for b, (correct, total) in sorted(bmap.items()):
            calibration[layer][b] = round(correct / total, 4) if total > 0 else None

    _CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_CALIBRATION_OUT, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Calibration written to {_CALIBRATION_OUT}")
    for layer, bmap in calibration.items():
        print(f"  {layer}:")
        for b, acc in sorted(bmap.items()):
            print(f"    conf {b}: {acc}")
    return calibration


def load_calibration() -> dict:
    """Load calibration.json if it exists; return empty dict otherwise."""
    try:
        if _CALIBRATION_OUT.exists():
            with open(_CALIBRATION_OUT, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def calibrated_confidence(layer: str, raw_conf: float, calibration: dict) -> float:
    """Map raw classifier confidence to calibrated value via bucket lookup."""
    b   = _bucket(raw_conf)
    val = calibration.get(layer, {}).get(b)
    return float(val) if val is not None else raw_conf


if __name__ == "__main__":
    run_calibration()
