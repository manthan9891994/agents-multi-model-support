"""Unit tests for PII scrubber."""

from classifier.infra.pii_scrubber import scrub


def test_clean_text_unchanged():
    r = scrub("What are contraindications for ACE inhibitors?")
    assert r.was_scrubbed is False
    assert r.matches == []
    assert "ACE inhibitors" in r.text


def test_mrn_scrubbed():
    r = scrub("Patient MRN: 12345678 has elevated creatinine.")
    assert r.was_scrubbed is True
    assert "[MRN]" in r.text
    assert "12345678" not in r.text
    assert "elevated creatinine" in r.text


def test_ssn_scrubbed():
    r = scrub("SSN 123-45-6789 should be removed.")
    assert "[SSN]" in r.text
    assert "123-45-6789" not in r.text


def test_dob_scrubbed():
    r = scrub("DOB: 03/14/1958")
    assert "[DOB]" in r.text
    assert "1958" not in r.text


def test_phone_scrubbed():
    r = scrub("Call 555-123-4567 for follow-up.")
    assert "[PHONE]" in r.text
    assert "555" not in r.text


def test_phone_with_parens_scrubbed():
    r = scrub("Office: (555) 123-4567")
    assert "[PHONE]" in r.text


def test_email_scrubbed():
    r = scrub("Email john.smith@hospital.org for results.")
    assert "[EMAIL]" in r.text
    assert "john.smith" not in r.text


def test_title_name_scrubbed():
    r = scrub("Patient seen by Dr. Smith yesterday.")
    assert "[NAME]" in r.text
    assert "Smith" not in r.text


def test_multiple_pii_types():
    text = "MRN: 87654321, Dr. Patel called patient at 555-987-6543 about jane@example.com"
    r = scrub(text)
    assert r.was_scrubbed is True
    assert {"[MRN]", "[NAME]", "[PHONE]", "[EMAIL]"}.issubset(set(r.matches))
    for sensitive in ["87654321", "Patel", "555-987-6543", "jane@example"]:
        assert sensitive not in r.text


def test_clinical_context_preserved():
    """Clinical terms should never be scrubbed."""
    r = scrub("Hemoglobin 6.8, MCV 72, reticulocyte 0.8% — interpret this anemia.")
    assert r.was_scrubbed is False
    assert "Hemoglobin" in r.text
    assert "anemia" in r.text


def test_strict_mode_catches_caps_names():
    text = "Patient JOHN SMITH presents with chest pain."
    r_loose = scrub(text, strict=False)
    r_strict = scrub(text, strict=True)
    assert r_loose.was_scrubbed is False
    assert r_strict.was_scrubbed is True
    assert "[NAME]" in r_strict.text
    assert "JOHN SMITH" not in r_strict.text


def test_strict_mode_catches_addresses():
    r = scrub("Patient lives at 123 Main Street.", strict=True)
    assert "[ADDRESS]" in r.text
    assert "Main Street" not in r.text


def test_empty_input():
    r = scrub("")
    assert r.was_scrubbed is False
    assert r.text == ""


def test_idempotent_scrub():
    """Scrubbing already-scrubbed text shouldn't introduce changes."""
    once = scrub("MRN: 12345678 needs follow-up")
    twice = scrub(once.text)
    assert twice.text == once.text


def test_layer2_uses_scrubber(monkeypatch):
    """L2 API call should scrub task before sending to Gemini."""
    from classifier.layers.layer2 import api as l2_api

    # Reset shared client + breaker so the mock is seen and breaker is closed.
    l2_api._shared_client = None
    l2_api._circuit_breaker._failures = 0
    l2_api._circuit_breaker._opened_at = 0.0
    captured = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        class models:
            @staticmethod
            def generate_content(model, contents, config):
                captured["contents"] = contents

                class FakeResp:
                    text = '{"task_type": "reasoning", "complexity": "simple", "confidence": 0.9, "reasoning": "test"}'

                return FakeResp()

    monkeypatch.setattr(l2_api.genai, "Client", FakeClient)

    l2_api._call_api("Patient MRN: 12345678 with chest pain")

    # Inspect what was actually sent — task should be scrubbed
    contents_str = str(captured["contents"])
    assert "12345678" not in contents_str
    assert "[MRN]" in contents_str
