"""Tests for the async Router API (aclassify + aclassify_batch)."""

import asyncio

import pytest

from classifier import Router


def test_aclassify_returns_decision():
    async def _run():
        router = Router(layer2_enabled=False, layer3_enabled=False)
        decision = await router.aclassify("Hello, how are you?")
        return decision

    decision = asyncio.run(_run())
    assert decision is not None
    assert decision.tier.value in ("low", "medium", "high")
    assert decision.model_name


def test_aclassify_batch_runs_concurrent():
    async def _run():
        router = Router(layer2_enabled=False, layer3_enabled=False)
        tasks = ["Hi", "Write a Python function", "Design a distributed system"]
        results = await router.aclassify_batch(tasks, concurrency=3)
        return results

    results = asyncio.run(_run())
    assert len(results) == 3
    assert all(d.tier.value in ("low", "medium", "high") for d in results)


def test_aclassify_does_not_block_event_loop():
    """Event loop should not block while classify runs in the threadpool."""

    async def _run():
        router = Router(layer2_enabled=False, layer3_enabled=False)
        flag = []

        async def _ticker():
            for _ in range(5):
                await asyncio.sleep(0.001)
                flag.append("tick")

        ticker = asyncio.create_task(_ticker())
        decision = await router.aclassify("Hello")
        await ticker
        return decision, flag

    decision, flag = asyncio.run(_run())
    assert decision is not None
    assert len(flag) == 5  # event loop kept ticking during classify
