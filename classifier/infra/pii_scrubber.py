"""PII scrubber — strips healthcare/personal identifiers before external LLM calls.

Replaces patterns with stable tokens so the L2 classifier can still understand
context (it sees `[MRN]` instead of "MRN: 12345678") without leaking PHI to Gemini.

Patterns covered:
  - MRN (medical record number)
  - SSN (XXX-XX-XXXX)
  - DOB (M/D/YYYY or MM/DD/YYYY)
  - Phone (US 10-digit, multiple formats)
  - Email
  - Names with title (Dr./Mr./Mrs./Ms. + capitalized word)

Strict mode also redacts:
  - All-caps name patterns (FIRSTNAME LASTNAME)
  - Address-like patterns
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Regex DoS guard — patterns that take longer than this on a single input are
# considered hostile and dropped. 1s is generous; well-formed PII patterns
# complete in microseconds.
_REGEX_TIMEOUT_SECS = 1.0

# ── Patterns ──────────────────────────────────────────────────────────────────
_MRN = re.compile(r"\b(?:MRN|mrn|Medical Record(?:\s+Number)?)[:\s#]*\d{6,12}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DOB = re.compile(r"\b(?:DOB|dob|D\.O\.B\.|Date of [Bb]irth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")  # generic date — caught after DOB
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TITLE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_CAPS_NAME = re.compile(r"\b[A-Z][A-Z]+\s+[A-Z][A-Z]+\b")  # JOHN DOE
_ADDRESS = re.compile(
    r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd)\b"
)


@dataclass
class ScrubResult:
    text: str
    was_scrubbed: bool
    matches: list[str] = field(default_factory=list)


# Extra patterns registered at runtime (e.g. via Router(extra_pii_patterns=...))
_extra_patterns: list[tuple] = []  # list of (compiled_regex, replacement_token)


def _validate_pattern(pat) -> bool:
    """Test the pattern against a small adversarial input under a timeout.
    Catches catastrophic backtracking before it gets near user data."""
    test_input = "a" * 50 + "!"
    runner_done = threading.Event()
    runner_ok = [False]

    def _run():
        try:
            pat.search(test_input)
            runner_ok[0] = True
        except Exception:
            pass
        finally:
            runner_done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    runner_done.wait(_REGEX_TIMEOUT_SECS)
    return runner_ok[0]


def register_extra_patterns(patterns: list[tuple]) -> None:
    """Register additional (regex, token) PII patterns. Idempotent on identity.

    Patterns are tested for catastrophic backtracking before registration —
    any pattern that doesn't return on a small input within the timeout is
    rejected with a logged warning.
    """
    for pat, tok in patterns:
        existing = any(p is pat for p, _ in _extra_patterns)
        if existing:
            continue
        if not _validate_pattern(pat):
            logger.warning(
                "pii_scrubber: rejected pattern %r — exceeded %.1fs timeout (possible regex DoS)",
                getattr(pat, "pattern", pat),
                _REGEX_TIMEOUT_SECS,
            )
            continue
        _extra_patterns.append((pat, tok))


def clear_extra_patterns() -> None:
    """Test helper — wipe registered extra patterns."""
    _extra_patterns.clear()


def scrub(text: str, strict: bool = False) -> ScrubResult:
    """Replace PII patterns with stable tokens. Returns scrubbed text + audit info.

    Args:
        text:   Input string to scrub.
        strict: If True, also scrubs all-caps names and address patterns.

    Returns:
        ScrubResult(text=scrubbed, was_scrubbed=bool, matches=[token,...])
    """
    if not text:
        return ScrubResult(text="", was_scrubbed=False)

    matches: list[str] = []
    out = text

    # Order matters — DOB before generic date, MRN before bare numbers
    for pat, token in [
        (_MRN, "[MRN]"),
        (_SSN, "[SSN]"),
        (_DOB, "[DOB]"),
        (_DATE, "[DATE]"),
        (_PHONE, "[PHONE]"),
        (_EMAIL, "[EMAIL]"),
        (_TITLE, "[NAME]"),
    ]:
        if pat.search(out):
            matches.append(token)
            out = pat.sub(token, out)

    if strict:
        for pat, token in [(_CAPS_NAME, "[NAME]"), (_ADDRESS, "[ADDRESS]")]:
            if pat.search(out):
                if token not in matches:
                    matches.append(token)
                out = pat.sub(token, out)

    # Apply extra user-registered patterns
    for pat, token in _extra_patterns:
        if pat.search(out):
            if token not in matches:
                matches.append(token)
            out = pat.sub(token, out)

    return ScrubResult(text=out, was_scrubbed=bool(matches), matches=matches)
