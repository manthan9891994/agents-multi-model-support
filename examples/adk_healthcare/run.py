"""Demo runner — wires one of the example agents into ADK Runner and sends a few tasks.

Each task triggers `dynamic_model_selector` (from `classifier.integrations.adk`)
which classifies the task and overrides `llm_request.model` BEFORE the API call.

Run: python -m examples.adk_healthcare.run
"""
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Pick any of the four example agents
from examples.adk_healthcare.agent_clinical_qa import root_agent


async def run_task(runner: Runner, session_service: InMemorySessionService, task: str):
    session = await session_service.create_session(app_name="demo", user_id="u1")
    message = types.Content(role="user", parts=[types.Part.from_text(text=task)])

    print(f"\n{'='*60}")
    print(f"  TASK: {task[:60]}")
    print(f"{'='*60}")

    try:
        async for _ in runner.run_async(user_id="u1", session_id=session.id, new_message=message):
            pass
    except Exception as exc:
        print(f"  (API error — see logs: {exc!s:.80})")


async def main():
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="demo", session_service=session_service)

    tasks = [
        "What are contraindications for ACE inhibitors?",
        "A 65-year-old presents with chest pain. What diagnostic workup is needed?",
        "Design a comprehensive cardiovascular risk reduction strategy for a 58-year-old "
        "with metabolic syndrome, hypertension, dyslipidemia, and family history of early MI.",
    ]
    for task in tasks:
        await run_task(runner, session_service, task)


if __name__ == "__main__":
    asyncio.run(main())
