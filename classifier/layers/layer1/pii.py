import re

from classifier.config.feature_flags import feature_flags

_PII_PATTERNS = {
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "email":       re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone":       re.compile(r"\b\+?1?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    "api_key":     re.compile(r"\b(sk-|pk_|AIza|ghp_|xox[baprs]-)[A-Za-z0-9_-]{16,}"),
    "jwt":         re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "mrn":         re.compile(r"\bMRN[\s:]*\d{4,}\b", re.IGNORECASE),
    "dob":         re.compile(r"\bDOB[\s:]*\d{4}-\d{2}-\d{2}\b", re.IGNORECASE),
}


def detect_pii(text: str) -> bool:
    """Return True if any PII/secret pattern is found. Only runs when pii_detection flag is on."""
    if not feature_flags.pii_detection:
        return False
    for pat in _PII_PATTERNS.values():
        if pat.search(text):
            return True
    return False
