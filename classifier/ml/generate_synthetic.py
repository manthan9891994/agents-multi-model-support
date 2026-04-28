"""Generate synthetic labeled training data using Gemini.

Usage:
    python -m classifier.ml.generate_synthetic                     # default: 30 per slot
    python -m classifier.ml.generate_synthetic --per-slot 50       # custom
    python -m classifier.ml.generate_synthetic --domain healthcare # domain-flavored output

Writes to: classifier/data/synthetic_tasks.jsonl (appends)

Two-pass generation per slot:
  Pass A (60%): generic phrasings, domain-agnostic
  Pass B (40%): domain-flavored (e.g., healthcare)
This balances breadth (so the classifier doesn't over-fit to clinical vocab)
with depth (so domain-heavy production traffic isn't a distribution shift).

Cost: ~$0.05 for 1,800 examples using gemini-2.5-flash-lite.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from classifier.core.types import TaskType, TaskComplexity
from classifier.infra.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_OUT_FILE = Path(__file__).parent.parent / "data" / "synthetic_tasks.jsonl"

# Skip slots that don't make sense (CONVERSATION is always SIMPLE)
_SKIP_SLOTS = {
    (TaskType.CONVERSATION, TaskComplexity.STANDARD),
    (TaskType.CONVERSATION, TaskComplexity.COMPLEX),
    (TaskType.CONVERSATION, TaskComplexity.RESEARCH),
}

_GENERIC_PROMPT = """Generate {count} diverse user task prompts.

Constraints:
- task_type:  {tt}
- complexity: {cx}
- {complexity_hint}
- Vary sentence structure (questions, imperatives, fragments)
- Avoid generating the same task with minor word changes
- Return ONLY a JSON array of strings, no commentary"""

_DOMAIN_PROMPT = """Generate {count} diverse user task prompts for a {domain} agent.

Constraints:
- task_type:  {tt}
- complexity: {cx}
- {complexity_hint}
- Use real {domain} vocabulary: {vocab}
- Vary sentence structure
- Return ONLY a JSON array of strings, no commentary"""

_COMPLEXITY_HINTS = {
    TaskComplexity.SIMPLE:   "5–20 words, single-clause request",
    TaskComplexity.STANDARD: "20–50 words, structured but not deep",
    TaskComplexity.COMPLEX:  "40–100 words, multi-part requirements",
    TaskComplexity.RESEARCH: "80–200 words, expert-level depth or multi-stage planning",
}

_DOMAIN_VOCAB = {
    "healthcare": (
        "prior authorization, ICD-10, FHIR, EHR, clinical notes, medication "
        "reconciliation, differential diagnosis, lab interpretation, discharge "
        "summary, patient education, drug interaction, treatment plan"
    ),
    "fintech": (
        "portfolio, derivative, hedging, risk-adjusted return, options pricing, "
        "compliance filing, KYC, AML, market data, position sizing"
    ),
    "legal": (
        "contract clause, NDA, non-compete, indemnification, liability, "
        "case citation, statutory interpretation, motion, deposition"
    ),
}


def _build_prompt(tt: TaskType, cx: TaskComplexity, count: int, domain: str | None) -> str:
    hint = _COMPLEXITY_HINTS[cx]
    if domain and domain in _DOMAIN_VOCAB:
        return _DOMAIN_PROMPT.format(
            count=count, tt=tt.value, cx=cx.value,
            complexity_hint=hint, domain=domain, vocab=_DOMAIN_VOCAB[domain],
        )
    return _GENERIC_PROMPT.format(
        count=count, tt=tt.value, cx=cx.value, complexity_hint=hint,
    )


def _call_gemini(prompt: str, model: str) -> list[str]:
    """One Gemini call → list of task strings. Returns [] on any failure."""
    from google import genai
    client = genai.Client(api_key=settings.google_api_key)
    cfg = genai.types.GenerateContentConfig(
        temperature=0.95,
        response_mime_type="application/json",
        max_output_tokens=4000,
    )
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        data = json.loads(resp.text)
        if isinstance(data, list):
            return [str(s).strip() for s in data if isinstance(s, str) and s.strip()]
        logger.warning("non-list response: %.80s", resp.text)
        return []
    except Exception as exc:
        logger.warning("gemini call failed: %s", exc)
        return []


def generate_slot(
    tt: TaskType, cx: TaskComplexity, count: int, domain: str | None, model: str,
) -> list[dict]:
    """Generate examples for a single (task_type, complexity) slot."""
    if (tt, cx) in _SKIP_SLOTS:
        return []
    generic_count = int(count * 0.6)
    domain_count  = count - generic_count if domain else 0
    if not domain:
        generic_count = count

    rows: list[dict] = []
    for n, dom in [(generic_count, None), (domain_count, domain)]:
        if n <= 0:
            continue
        prompt = _build_prompt(tt, cx, n, dom)
        tasks = _call_gemini(prompt, model)
        for t in tasks:
            rows.append({"task": t, "task_type": tt.value, "complexity": cx.value})
    logger.info("  %s/%s: %d examples", tt.value, cx.value, len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic L3 training data")
    parser.add_argument("--per-slot", type=int, default=30, help="examples per (type, complexity) slot")
    parser.add_argument("--domain", type=str, default="", help="optional domain flavoring (healthcare/fintech/legal)")
    parser.add_argument("--model",  type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--out",    type=str, default=str(_OUT_FILE))
    args = parser.parse_args()

    domain = args.domain.strip().lower() or None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY not configured. Set it in .env first.")

    total_rows: list[dict] = []
    for tt in TaskType:
        for cx in TaskComplexity:
            rows = generate_slot(tt, cx, args.per_slot, domain, args.model)
            total_rows.extend(rows)

    # Append (don't overwrite) so multiple runs with different domains accumulate
    with open(out_path, "a", encoding="utf-8") as f:
        for row in total_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d synthetic examples → %s", len(total_rows), out_path)


if __name__ == "__main__":
    main()
