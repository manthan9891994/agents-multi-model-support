"""Load labeled training data from reference_tasks.jsonl + synthetic supplements.

Data format (one JSON object per line):
    {"task": "...", "task_type": "code_creation", "complexity": "standard"}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_REFERENCE_FILE = _DATA_DIR / "reference_tasks.jsonl"
_SYNTHETIC_FILE = _DATA_DIR / "synthetic_tasks.jsonl"


def load_examples(
    include_synthetic: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Return (texts, task_types, complexities) parallel lists from labeled JSONL."""
    sources = [_REFERENCE_FILE]
    if include_synthetic and _SYNTHETIC_FILE.exists():
        sources.append(_SYNTHETIC_FILE)

    texts:        list[str] = []
    task_types:   list[str] = []
    complexities: list[str] = []

    for path in sources:
        if not path.exists():
            logger.warning("data_loader: file not found %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    texts.append(row["task"])
                    task_types.append(row["task_type"])
                    complexities.append(row["complexity"])
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug("data_loader: skipping malformed line in %s — %s", path.name, exc)

    logger.info(
        "data_loader: loaded %d examples (reference=%d, synthetic=%d)",
        len(texts),
        sum(1 for p in [_REFERENCE_FILE] if p.exists()),
        sum(1 for p in [_SYNTHETIC_FILE] if p.exists()) if include_synthetic else 0,
    )
    return texts, task_types, complexities
