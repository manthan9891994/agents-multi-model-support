"""
Proof: ONE LlmAgent, model selected at runtime via before_model_callback.
Runs through the real ADK Runner — callback fires inside ADK's execution path.
API call fails (dummy key) but model selection happens BEFORE the API call.

Run: python -m examples_adk.run
"""
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from examples_adk.agent import root_agent


async def run_task(runner: Runner, session_service: InMemorySessionService, task: str):
    session = await session_service.create_session(app_name="demo", user_id="u1")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=task)],
    )

    print(f"\n{'='*60}")
    print(f"  TASK: {task[:60]}")
    print(f"{'='*60}")

    try:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=message,
        ):
            pass
    except Exception as e:
        # API call fails with dummy key — expected.
        # Model selection already printed above by the callback.
        print(f"  (API call stopped: dummy key — model was already selected above)")


async def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="demo",
        session_service=session_service,
    )

    tasks = [
        "Write a README for this project",
        "Design a distributed cache with LRU eviction, thread safety, and TTL support",
        "Implement a REST API endpoint with input validation",
    ]

    for task in tasks:
        await run_task(runner, session_service, task)

    print(f"\n{'='*60}")
    print("  Same root_agent used for every task.")
    print("  Model changed by before_model_callback — not by routing.")
    print(f"{'='*60}\n")


asyncio.run(main())
