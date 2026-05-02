"""Concurrency stress test — verify Router is thread-safe under load."""
import concurrent.futures
import threading

import pytest

from classifier import Router


def test_50_concurrent_classifies():
    """50 concurrent calls should all succeed without deadlock or corrupted state."""
    router = Router(layer2_enabled=False, layer3_enabled=False)
    tasks = [f"task number {i}" for i in range(50)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(router.classify, tasks))

    assert len(results) == 50
    assert all(d is not None for d in results)
    assert all(d.tier.value in ("low", "medium", "high") for d in results)


def test_concurrent_routers_with_different_overrides():
    """Two Router instances with different settings shouldn't leak state."""
    r_layer1_only = Router(layer2_enabled=False, layer3_enabled=False)
    r_default = Router()

    barrier = threading.Barrier(2)
    results = {}

    def call(tag, router):
        barrier.wait()
        for _ in range(20):
            d = router.classify("hello")
            results.setdefault(tag, []).append(d.layer_used)

    t1 = threading.Thread(target=call, args=("a", r_layer1_only))
    t2 = threading.Thread(target=call, args=("b", r_default))
    t1.start(); t2.start()
    t1.join();  t2.join()

    # r_layer1_only should never use L2 or L3
    assert all(layer == "layer1" for layer in results.get("a", []))
