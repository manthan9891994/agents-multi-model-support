# Quickstart

## Install

```bash
pip install 'dynamic-model-router[ml]'
```

## Minimal usage

```python
from classifier import Router

router = Router()
decision = router.classify("Summarise this contract")

print(decision.tier.value)       # "medium"
print(decision.model_name)       # "gemini-2.5-flash"
print(decision.layer_used)       # "layer1" / "layer2" / "layer3"
print(decision.confidence)       # 0.0 - 1.0
```

## Estimate cost before calling

```python
info = router.estimate_cost("Design a distributed system architecture")
print(info["est_usd_per_call"])  # 0.0000125
```

## Async (FastAPI / aiohttp)

```python
decision = await router.aclassify("Translate this to French")
results = await router.aclassify_batch(["task1", "task2", "task3"])
```

## Decorator API

```python
from classifier import route_model

@route_model(provider="anthropic")
def call_llm(task: str, model_name: str = "claude-sonnet-4-6"):
    # model_name auto-injected per-task
    ...
```

## CLI

```bash
dmr classify "Design a binary search tree"
dmr eval --data my_test.jsonl
dmr benchmark
dmr doctor
dmr version
```
