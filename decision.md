# Dynamic Model Selection for Multi-Agent Frameworks
## Feasibility Analysis & Implementation Decisions

**Date**: 2026-04-24  
**Status**: FEASIBLE - Ready for MVP Implementation  
**Priority**: HIGH (Cost Optimization Opportunity)

---

## Executive Summary

✅ **VERDICT: YES, THIS IS FEASIBLE AND PRACTICAL**

Dynamic model selection across Google ADK and CrewAI is achievable in Python. The key insight from the discussion is:
- Don't swap models inside a running agent (hard, unreliable)
- **Route at the orchestration layer** using a central decision engine
- Keep the same model throughout an iteration (consistency)
- Switch models between requests based on task classification

**Estimated Implementation Effort**: 2-3 weeks for MVP  
**Cost Savings Potential**: 40-60% on API costs (depending on task mix)

---

## Architecture Decision: 3-Tier Recommendation

### Option A: MVP - External Routing Layer ⭐ **RECOMMENDED**
**Timeline**: 1-2 weeks | **Risk**: Low | **Flexibility**: High

```
Request → Task Classifier → Model Router → Agent Selector → Execute
         (categorize)      (choose model)  (which agent)
```

**Implementation**:
- Central `ModelRouter` class evaluates request + runtime state
- Task classification: simple/reasoning/coding/research
- Thin adapters for ADK and CrewAI
- Models organized in 3 tiers: light/standard/heavy

**Pros**:
- Easiest to build and test
- Clear decision logging (why model was chosen)
- Works with both frameworks without deep SDK changes
- Easy to add fallback logic

**Cons**:
- Requires orchestration layer above agents
- Not transparent to existing code

**Best For**: Getting to market quickly, clear cost tracking

---

### Option B: Unified Router + Adapters
**Timeline**: 2-3 weeks | **Risk**: Medium | **Flexibility**: Very High

Add standardized adapter pattern:
```python
RouterPolicy + ModelRegistry + RouteDecision
     ↓
ADKAgentAdapter + OpenAIAgentAdapter
     ↓
CrewAIAdapter + other-frameworks
```

**Pros**:
- Highly extensible (add new frameworks easily)
- Reusable routing logic
- Better for production package

**Cons**:
- More upfront design work
- More code to maintain

**Best For**: Building a product to share/monetize

---

### Option C: In-SDK Model Indirection
**Timeline**: 3-4 weeks | **Risk**: High | **Flexibility**: Medium

- CrewAI: Custom agent class that decides model per step
- ADK: Use callbacks + custom orchestration
- Both: Custom model provider/connector pattern

**Pros**:
- Closer to "transparent" behavior
- No orchestration wrapper needed

**Cons**:
- Tight coupling to SDK versions
- Tool calling differences between models can break
- Harder to test and debug

**Best For**: v2 when you have real usage data

---

## Framework Capability Matrix

| Capability | CrewAI | Google ADK | Notes |
|---|---|---|---|
| **Multiple models per agent** | ✅ Yes (Agent.model param) | ✅ Yes (model param) | Both support this natively |
| **Model selection at runtime** | ⚠️ Per-task only | ⚠️ Limited | Neither does mid-execution swap |
| **Custom model providers** | ✅ Callbacks + inheritance | ✅ Callbacks + custom agents | Good extension points |
| **Cost tracking** | Manual | Manual | Need custom logging |
| **Fallback/retry with alt model** | ✅ Can build on callbacks | ✅ Can build on callbacks | Good extension surface |
| **Streaming support** | ✅ Yes | ✅ Yes | Both support streaming |
| **Tool calling consistency** | ⚠️ Model-dependent | ⚠️ Model-dependent | **Risk**: different models may fail at tools |

**Key Risk**: Tool calling behavior varies by model family.  
**Mitigation**: Test tool compatibility matrix for your chosen models before production.

---

## Task Classification Strategy

How to route requests:

```python
class TaskComplexity(Enum):
    SIMPLE = "light"        # < 500 tokens of context
    STANDARD = "standard"   # 500-5K tokens
    COMPLEX = "heavy"       # > 5K tokens OR needs reasoning
    RESEARCH = "research"   # Long-context, multi-step

class ModelRouter:
    def classify_request(self, request: str, context: dict) -> TaskComplexity:
        # Heuristics: token count, keyword detection, data volume
        pass
    
    def choose_model(self, task: TaskComplexity, agent_role: str) -> str:
        # Returns: "gpt-4o-mini" | "gpt-4" | "claude-3-haiku" | "claude-3-opus"
        pass
```

**Example routing table**:
```
SIMPLE task → cheaper/faster model (e.g., Claude 3.5 Haiku or GPT-4o Mini)
STANDARD task → balanced model (e.g., Claude 3.5 Sonnet or GPT-4)
COMPLEX task → heavyweight (e.g., Claude Opus or GPT-4 Turbo)
RESEARCH task → longest-context (e.g., Claude Opus with extended context)
```

---

## Critical Design Constraints

### ✅ Keep the Same Model Throughout One Iteration
**Solution**: Store model choice in execution context
```python
execution_context = {
    "chosen_model": "gpt-4o-mini",
    "task_complexity": TaskComplexity.SIMPLE,
    "classified_at": timestamp
}
# All steps in this execution use this model
```

### ✅ Change Models Between Requests
**Solution**: Router re-evaluates on each new request
```python
def process_request(request):
    decision = router.decide(request)  # Fresh decision each time
    agent.model = decision.model_name
    return agent.execute()
```

### ✅ Fast Decision Making
**Solution**: Lightweight classification engine
- Don't call LLM to classify (too slow)
- Use token counting + heuristics
- Cache classification results if possible
- Target: < 100ms decision time

### ✅ Innovative Approach
**Ideas**:
1. **Request Fingerprinting**: Hash request to reuse classification
2. **Adaptive Routing**: Learn from actual execution time which model was best
3. **Cost-Aware Selection**: Include cost/latency targets in routing (not just task type)
4. **Similarity Clustering**: Use embeddings to find similar past requests

---

## Implementation Roadmap

### Phase 1: MVP (Week 1-2) ⭐
- [ ] Define 3 model tiers and example models
- [ ] Implement `TaskClassifier` (token count + keywords)
- [ ] Implement `ModelRouter` class
- [ ] Implement `RouterDecision` dataclass
- [ ] Adapter: CrewAI integration
- [ ] Adapter: Google ADK integration
- [ ] Basic logging (why was this model chosen?)
- [ ] Unit tests for routing logic

**Deliverable**: Working Python package you can import

### Phase 2: Production Ready (Week 3)
- [ ] Add fallback logic (if model fails, retry with stronger model)
- [ ] Add cost tracking per request/agent
- [ ] Add latency monitoring
- [ ] Configuration schema (YAML/JSON for routing policies)
- [ ] CLI tool: `model-router config` + `model-router test`
- [ ] Documentation + examples

**Deliverable**: Ready for production use

### Phase 3: Advanced Features (Week 4+)
- [ ] Adaptive learning (track actual task difficulty)
- [ ] Request fingerprinting + caching
- [ ] Cost/latency optimization reports
- [ ] Multi-provider balancing (Anthropic + OpenAI + local LLMs)

---

## Code Structure (Proposed)

```
model-router-py/
├── model_router/
│   ├── __init__.py
│   ├── core.py                    # ModelRouter, RouteDecision, TaskClassifier
│   ├── policies.py                # RoutingPolicy, model selection logic
│   ├── adapters/
│   │   ├── crewai_adapter.py      # CrewAI integration
│   │   └── adk_adapter.py         # Google ADK integration
│   ├── models.py                  # Data classes
│   ├── logging.py                 # Structured logging
│   └── utils.py                   # Token counting, helpers
├── tests/
│   ├── test_router.py
│   ├── test_classifiers.py
│   └── test_adapters.py
├── examples/
│   ├── crewai_example.py
│   ├── adk_example.py
│   └── config.yaml
├── README.md
└── requirements.txt
```

---

## Known Challenges & Mitigations

| Challenge | Severity | Mitigation |
|---|---|---|
| **Tool calling differences between models** | 🔴 High | Pre-test tool compatibility matrix; add adapter layer if needed |
| **Model availability changes** | 🟡 Medium | Configuration-driven model mappings; fallback chain |
| **Structured output format changes** | 🟡 Medium | Detect model in output normalization; validate schema |
| **Context window mismatches** | 🟡 Medium | Measure input before selecting model |
| **Token counting accuracy** | 🟠 Low | Use tiktoken for OpenAI, claude-tokenizer for Anthropic |
| **Latency variance** | 🟠 Low | Cache metrics per model; update routing based on SLAs |

---

## Quick Decision Checklist

Before building, confirm:
- [ ] Team agrees on 3-5 target models (don't support too many)
- [ ] Tool compatibility is acceptable across chosen models
- [ ] Cost savings target is clear (e.g., "reduce 40% of API spend")
- [ ] Logging/monitoring requirements defined
- [ ] Whether this is MVP or production-grade from day 1

---

## Next Steps

1. **Validate**: Confirm the 3-5 models you want to support
2. **Test**: Clone repos, run quick compatibility test (tool calling across models)
3. **Design**: Define exact routing policy (task → model mapping)
4. **Build**: Start with Phase 1 MVP (ModelRouter + one adapter)
5. **Deploy**: Measure actual cost savings & iterate

---

## References

- **Discussion**: See `discussion.md` for detailed architectural analysis
- **CrewAI**: https://github.com/crewaiinc/crewai
- **Google ADK**: https://github.com/google/adk-python

---

**Decision Made**: Proceed with **Option A (MVP External Routing Layer)**  
**Confidence Level**: ⭐⭐⭐⭐⭐ (Very High)  
**Ready to implement**: YES
