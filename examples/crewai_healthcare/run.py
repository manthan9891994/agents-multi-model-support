"""CrewAI demo — same crew handles tasks of varying complexity, each routed to
the right model.

Run: python -m examples.crewai_healthcare.run
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from crewai import Agent, Crew, Task

from classifier.integrations.crewai import pick_llm_for_task


# ── Three healthcare tasks of different complexity ────────────────────────────
TASKS = [
    "What are the contraindications for ACE inhibitors?",
    "A 65-year-old presents with chest pain and shortness of breath. What diagnostic workup is needed?",
    "Design a comprehensive cardiovascular risk reduction strategy for a 58-year-old "
    "with metabolic syndrome, hypertension, dyslipidemia, and family history of early MI. "
    "Include lifestyle modifications, pharmacotherapy, monitoring schedule, and drug-drug interaction considerations.",
]


def build_crew_for_task(task_text: str) -> Crew:
    """Construct a single-agent crew with the LLM picked by the router."""
    llm = pick_llm_for_task(task_text)

    researcher = Agent(
        role="Clinical Researcher",
        goal="Provide accurate, evidence-based clinical answers",
        backstory=(
            "You are a board-certified internal medicine physician with deep "
            "knowledge of cardiology, endocrinology, and pharmacology."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=task_text,
        expected_output="A clinically accurate, well-structured answer.",
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], verbose=False)


def main():
    for task_text in TASKS:
        print(f"\n{'='*70}")
        print(f"  TASK: {task_text[:80]}")
        print(f"{'='*70}")
        crew = build_crew_for_task(task_text)
        try:
            result = crew.kickoff()
            print(f"  → Result preview: {str(result)[:200]}...")
        except Exception as exc:
            print(f"  → ({exc!s:.100})")


if __name__ == "__main__":
    main()
