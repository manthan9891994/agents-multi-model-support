"""Unit tests for the KeywordPack builder API."""
import pytest

from classifier import KeywordPack, TaskType, ModelTier
from classifier.layers.layer1 import keyword_pack as kp


@pytest.fixture(autouse=True)
def reset():
    kp.clear_registered()
    yield
    kp.clear_registered()


def test_builder_basic():
    pack = (KeywordPack.builder("legal")
            .add(TaskType.REASONING, ["argue", "precedent"])
            .build())
    assert pack.name == "legal"
    assert TaskType.REASONING in pack.task_keywords
    assert "argue" in pack.task_keywords[TaskType.REASONING]["primary"]


def test_builder_chains_and_dedupes():
    pack = (KeywordPack.builder("test")
            .add(TaskType.REASONING, ["a", "b"])
            .add(TaskType.REASONING, ["b", "c"])  # b is duplicate
            .build())
    primary = pack.task_keywords[TaskType.REASONING]["primary"]
    assert primary == ["a", "b", "c"]


def test_builder_groups():
    pack = (KeywordPack.builder("test")
            .add(TaskType.REASONING, ["x"], group="primary")
            .add(TaskType.REASONING, ["y"], group="weak")
            .build())
    assert pack.task_keywords[TaskType.REASONING]["primary"] == ["x"]
    assert pack.task_keywords[TaskType.REASONING]["weak"]    == ["y"]


def test_escalator():
    pack = KeywordPack.builder("t").escalator("distributed", weight=2).build()
    assert pack.escalators["distributed"] == 2


def test_min_tier():
    pack = KeywordPack.builder("t").min_tier("hipaa", ModelTier.MEDIUM).build()
    assert pack.domain_min_tier["hipaa"] == ModelTier.MEDIUM


def test_register_extra_packs_idempotent():
    pack = KeywordPack.builder("once").add(TaskType.MATH, ["zztest"]).build()
    kp.register_extra_packs([pack])
    kp.register_extra_packs([pack])
    assert kp.list_registered() == ["once"]


def test_register_actually_modifies_l1_keywords():
    """Registering injects keywords into L1's _TASK_KEYWORDS dict."""
    from classifier.layers.layer1.constants import _TASK_KEYWORDS
    pack = KeywordPack.builder("inject").add(TaskType.REASONING, ["unique_xyzzy_kw"]).build()
    kp.register_extra_packs([pack])
    assert "unique_xyzzy_kw" in _TASK_KEYWORDS[TaskType.REASONING].get("primary", [])
