# FAQ

## Why do I see Layer 2 firing even though I disabled it?

You probably haven't disabled it everywhere. The order of precedence is:
1. `Router(layer2_enabled=False)` constructor argument
2. `LAYER2_ENABLED=false` env var (loaded by pydantic-settings)
3. Built-in default (`False`)

Constructor wins; env vars set the global. If you've enabled L2 globally and want one Router without it, pass the constructor argument.

## Why does my task always route to MEDIUM tier?

Likely PII detection. If the task contains anything matching MRN / SSN / DOB / phone / email patterns, the router forces tier ≥ MEDIUM and sets `decision.compliance_flag = True`. Check `decision.reasoning` — it'll contain "PII/PHI detected".

## Why does Layer 3 silently abstain?

Three common reasons (in order of likelihood):
1. **Model file missing**: `dmr doctor` shows `[!] L3 model file ... missing`. Run `dmr train --data classifier/data/synthetic_tasks.jsonl`.
2. **Confidence below threshold**: L3 returns `None` when its top prediction is below `layer3_confidence_threshold` (default 0.85). This is by design — under-confident L3 should fall through to L2.
3. **`sentence-transformers` not installed**: `pip install 'dynamic-model-router[ml]'`.

## How do I add my own keywords without forking?

```python
from classifier import Router, KeywordPack
from classifier.core.types import TaskType

pack = (KeywordPack.builder("legal")
        .add(TaskType.REASONING, ["indemnification", "precedent"])
        .build())
router = Router(extra_keyword_packs=[pack])
```

## How is this different from LiteLLM?

LiteLLM is a unified API client across providers. This package is a *classifier* that decides which model to use, then hands off to the provider's SDK. They're complementary — you can route with this package and call with LiteLLM.

## Does this need an internet connection?

Only Layer 2 does (it calls Gemini Flash Lite). Layers 1 and 3 are entirely local. Run with `Router(layer2_enabled=False)` for fully offline routing.

## Does this collect telemetry?

**No telemetry, ever.** The package never phones home. `dmr doctor` and `dmr stats` only read local files. If you want OpenTelemetry traces for your own observability stack, install `opentelemetry-api` — we'll emit spans into whatever exporter your application configured.
