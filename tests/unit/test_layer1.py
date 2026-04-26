"""Unit tests for classifier/layers/layer1.py — no external calls."""
import pytest
from classifier.layers.layer1 import classify_layer1
from classifier.core.types import ModelTier, TaskType, TaskComplexity
from classifier.core.exceptions import ClassificationError


# ── Error handling ─────────────────────────────────────────────────────────────

def test_empty_task_raises():
    with pytest.raises(ClassificationError):
        classify_layer1("")


def test_whitespace_task_raises():
    with pytest.raises(ClassificationError):
        classify_layer1("   ")


# ── Task-type routing ──────────────────────────────────────────────────────────

def test_readme_is_doc_creation():
    tt, _, _, _, _ = classify_layer1("Write a README for this project")
    assert tt == TaskType.DOC_CREATION


def test_palindrome_is_code_low():
    tt, _, tier, _, _ = classify_layer1("Write a function to check if a string is a palindrome")
    assert tt == TaskType.CODE_CREATION
    assert tier == ModelTier.LOW


def test_compare_is_reasoning():
    tt, _, _, _, _ = classify_layer1("Compare Python vs JavaScript for REST APIs")
    assert tt == TaskType.REASONING


def test_hello_is_conversation():
    tt, _, tier, _, _ = classify_layer1("Hello, how are you?")
    assert tt == TaskType.CONVERSATION
    assert tier == ModelTier.LOW


def test_calculate_is_math():
    tt, _, _, _, _ = classify_layer1("Calculate the eigenvalues of a 4x4 matrix")
    assert tt == TaskType.MATH


def test_translate_is_translation():
    tt, _, _, _, _ = classify_layer1("Translate this paragraph to Spanish")
    assert tt == TaskType.TRANSLATION


def test_image_is_multimodal():
    tt, _, _, _, _ = classify_layer1("Analyze this image and describe what you see")
    assert tt == TaskType.MULTIMODAL


# ── Complexity & tier routing ──────────────────────────────────────────────────

def test_simple_doc_routes_low():
    _, cx, tier, _, _ = classify_layer1("Write a README for this project")
    assert cx == TaskComplexity.SIMPLE
    assert tier == ModelTier.LOW


def test_distributed_cache_routes_high():
    _, _, tier, _, _ = classify_layer1(
        "Design a distributed cache with LRU eviction and thread safety"
    )
    assert tier == ModelTier.HIGH


def test_rest_api_routes_medium():
    _, _, tier, _, _ = classify_layer1("Implement a REST API endpoint with input validation")
    assert tier == ModelTier.MEDIUM


def test_research_routes_high():
    _, _, tier, _, _ = classify_layer1(
        "Comprehensive research on AI adoption across 10 industries with market data"
    )
    assert tier == ModelTier.HIGH


def test_simple_keyword_de_escalates():
    _, cx, _, _, _ = classify_layer1("Write a simple function to reverse a string")
    assert cx == TaskComplexity.SIMPLE


# ── Negation awareness ─────────────────────────────────────────────────────────

def test_negation_suppresses_code():
    tt, _, _, conf, _ = classify_layer1(
        "Don't write code, just explain how a binary search works"
    )
    # Should NOT be CODE_CREATION — negation suppresses it
    assert tt != TaskType.CODE_CREATION or conf < 0.6


# ── Algorithm names → COMPLEX minimum ─────────────────────────────────────────

def test_raft_forces_complex():
    _, cx, _, _, _ = classify_layer1("Implement the Raft consensus algorithm")
    assert cx in (TaskComplexity.COMPLEX, TaskComplexity.RESEARCH)


def test_bloom_filter_forces_complex():
    _, cx, _, _, _ = classify_layer1("Design a bloom filter for URL deduplication")
    assert cx in (TaskComplexity.COMPLEX, TaskComplexity.RESEARCH)


# ── Domain escalation ──────────────────────────────────────────────────────────

def test_hipaa_forces_high():
    _, _, tier, _, _ = classify_layer1(
        "Write a function to process patient data under HIPAA compliance"
    )
    assert tier == ModelTier.HIGH


def test_gdpr_forces_high():
    _, _, tier, _, _ = classify_layer1(
        "Design a data deletion flow for GDPR right-to-erasure compliance"
    )
    assert tier == ModelTier.HIGH


def test_clinical_forces_medium_minimum():
    _, _, tier, _, _ = classify_layer1("Write a summary of this clinical trial report")
    assert tier in (ModelTier.MEDIUM, ModelTier.HIGH)


# ── Format requests — suppress escalation ─────────────────────────────────────

def test_format_request_does_not_escalate():
    # "as json" should not escalate complexity beyond what content warrants
    _, cx, _, _, _ = classify_layer1("List the top 5 cities, return json")
    assert cx in (TaskComplexity.SIMPLE, TaskComplexity.STANDARD)


# ── Question type → SIMPLE ─────────────────────────────────────────────────────

def test_yes_no_question_is_simple():
    _, cx, _, _, _ = classify_layer1("Can you explain what a REST API is?")
    assert cx == TaskComplexity.SIMPLE


def test_what_is_question_is_simple():
    _, cx, _, _, _ = classify_layer1("What is a microservice architecture?")
    assert cx == TaskComplexity.SIMPLE


# ── Ambiguity detection ────────────────────────────────────────────────────────

def test_ambiguous_task_has_low_confidence():
    # "compare" hits REASONING and "implement" hits CODE_CREATION equally (3pts each)
    _, _, _, conf, _ = classify_layer1("Compare approaches and implement the best solution")
    assert conf <= 0.50  # tied scores → ambiguity → confidence capped at 0.45


# ── Conversation history bias ──────────────────────────────────────────────────

def test_history_biases_code_creation():
    history = [
        "Write a function to sort a list",
        "Now implement a binary search",
        "Debug this code for me",
    ]
    tt, _, _, _, _ = classify_layer1("Make it faster", history=history)
    assert tt == TaskType.CODE_CREATION


# ── Return type contract ───────────────────────────────────────────────────────

def test_return_types_are_correct():
    tt, cx, tier, conf, reason = classify_layer1("Write a README")
    assert isinstance(tt, TaskType)
    assert isinstance(cx, TaskComplexity)
    assert isinstance(tier, ModelTier)
    assert 0.0 <= conf <= 1.0
    assert isinstance(reason, str) and len(reason) > 0


def test_latency_is_fast():
    import time
    t0 = time.perf_counter()
    classify_layer1("Write a function to sort a list")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100  # Layer 1 must be < 100ms even without tiktoken


# ── Phase 3: Trivial-input guard (Item 18) ────────────────────────────────────

def test_trivial_single_char_is_conversation():
    tt, cx, tier, conf, _ = classify_layer1("k")
    assert tt == TaskType.CONVERSATION
    assert cx == TaskComplexity.SIMPLE
    assert tier == ModelTier.LOW
    assert conf >= 0.9


def test_trivial_emoji_only_is_conversation():
    tt, cx, tier, _, _ = classify_layer1("👍")
    assert tt == TaskType.CONVERSATION
    assert tier == ModelTier.LOW


def test_trivial_ack_word_is_conversation():
    tt, _, _, conf, _ = classify_layer1("okay")
    assert tt == TaskType.CONVERSATION
    assert conf >= 0.9


def test_non_trivial_short_task_classified_normally():
    tt, _, _, _, _ = classify_layer1("Write a README")
    assert tt == TaskType.DOC_CREATION


# ── Phase 3: PII detection (Item 1) ───────────────────────────────────────────

def test_detect_pii_email():
    from classifier.layers.layer1 import detect_pii
    assert detect_pii("Send results to john.doe@example.com") is True


def test_detect_pii_ssn():
    from classifier.layers.layer1 import detect_pii
    assert detect_pii("Patient SSN is 123-45-6789") is True


def test_detect_pii_api_key():
    from classifier.layers.layer1 import detect_pii
    assert detect_pii("My key: sk-abcdefghijklmnopqrstuvwx") is True


def test_detect_pii_mrn():
    from classifier.layers.layer1 import detect_pii
    assert detect_pii("Patient MRN: 48210 reports chest pain") is True


def test_no_pii_in_clean_task():
    from classifier.layers.layer1 import detect_pii
    assert detect_pii("Write a function to sort a list") is False


# ── Phase 3: Continuation detection (Item 13) ─────────────────────────────────

def test_continuation_inherits_history_type():
    history = ["implement a REST API endpoint", "add authentication to it"]
    tt, _, _, _, reason = classify_layer1("now make it faster", history=history)
    assert tt == TaskType.CODE_CREATION
    assert "continuation" in reason


def test_non_continuation_classified_normally():
    # "Write a README" doesn't start with a continuation word → classified by content, not history
    tt, _, _, _, reason = classify_layer1("Write a README for this project")
    assert tt == TaskType.DOC_CREATION
    assert "continuation" not in reason


# ── Phase 3: Provider tokenizer (Item 14) ─────────────────────────────────────

def test_provider_param_accepted():
    for provider in ("google", "anthropic", "openai"):
        result = classify_layer1("Write a function", provider=provider)
        assert len(result) == 5
