# ADK Healthcare Example Agents

Four ADK `LlmAgent` instances demonstrating how to wire `dynamic-model-router`
into Google ADK so the model is selected per request based on task complexity.

| Agent | Demonstrates |
|---|---|
| `ClinicalQAAgent`    | Wide tier spread (LOW → HIGH) within one session |
| `PriorAuthAgent`     | L3 abstain → L2 fallback path, tool calls (formulary lookup, PA submission) |
| `LabAnalyzerAgent`   | Multimodal context signal, math/reasoning routing |
| `ClinicalNoteAgent`  | PII compliance flag forces MEDIUM tier minimum |

## Setup

```bash
pip install 'dynamic-model-router[adk,ml]'
```

Set your Google API key in the project root `.env` file:

```
GOOGLE_API_KEY=your_key_here
LAYER3_ENABLED=true
LAYER3_STRATEGY=head
```

## Run via ADK Web (recommended)

```bash
adk web examples/adk_healthcare --port 8080
```

Then open http://127.0.0.1:8080 — pick an agent from the dropdown and chat.

Watch the console output: each request logs which model was selected and why.

## Run via the simple runner

```bash
python -m examples.adk_healthcare.run
```

## How the routing works

Each agent uses `before_model_callback=dynamic_model_selector` from the package.
The callback intercepts every LLM API call, classifies the user's task via the
3-layer cascade, and overwrites `llm_request.model` with the selected model
*before* the API call goes out.

```python
from google.adk.agents import LlmAgent
from classifier.integrations.adk import dynamic_model_selector

agent = LlmAgent(
    name="MyAgent",
    model="gemini-2.5-flash",   # placeholder — replaced per-request
    before_model_callback=dynamic_model_selector,
)
```

That's the entire integration surface. Every other line in these agents is
just defining their domain-specific instructions and tools.
