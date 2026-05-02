# CrewAI Healthcare Example

One CrewAI `Agent` whose LLM is selected per-task by the dynamic router.

## Setup

```bash
pip install crewai 'dynamic-model-router[ml]'
```

Set provider key in `.env`:

```
GOOGLE_API_KEY=your_key_here
```

## Run

```bash
python -m examples.crewai_healthcare.run
```

You'll see three tasks routed to different tiers:
- "What are contraindications for ACE inhibitors?" → LOW
- "65-year-old with chest pain..." → MEDIUM
- "Design comprehensive CV risk reduction..." → HIGH

## Two integration patterns

**Pattern A — Pick LLM per-task (this example):**

```python
from classifier.integrations.crewai import pick_llm_for_task

llm = pick_llm_for_task("Compare metformin vs GLP-1 agonists for T2DM")
agent = Agent(role="Researcher", goal="...", llm=llm)
```

Use when each task constructs its own crew.

**Pattern B — `DynamicLLM` wrapper (one agent, all tiers):**

```python
from classifier.integrations.crewai import DynamicLLM
from crewai import Agent

agent = Agent(role="Researcher", goal="...", llm=DynamicLLM())
# Same agent reused for all tasks; LLM picks model on each call.
```

Use when one Crew handles many tasks and you want uniform agent definitions.
