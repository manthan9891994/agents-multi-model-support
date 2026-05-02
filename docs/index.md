# dynamic-model-router

Routes each LLM call to the cheapest model that can handle it — before any API call is made.

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
configuration
faq
```

```{toctree}
:maxdepth: 2
:caption: Integrations

integrations/adk
integrations/crewai
integrations/langchain
integrations/autogen
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/router
api/decision
api/keyword_pack
api/cli
```

```{toctree}
:maxdepth: 1
:caption: Project

../CHANGELOG
../CONTRIBUTING
../SECURITY
../ROADMAP
```

## Why this exists

Most agent frameworks pin one model per agent. That's wasteful — a "Hello, how are you?" task gets the same `gpt-4` as "Design a distributed consensus algorithm." This package classifies each task in `<15ms` and picks the right tier.

## Install

```bash
pip install dynamic-model-router
# or with ML layer:
pip install 'dynamic-model-router[ml]'
```

## Five-line quickstart

```python
from classifier import classify
decision = classify("Design a CQRS architecture for healthcare records")
print(decision.tier)        # high
print(decision.model_name)  # gemini-2.5-pro
```
