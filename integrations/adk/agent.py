import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

from classifier import classify_task
from classifier.core.exceptions import ClassificationError
from classifier.infra.config import settings

logger = logging.getLogger(__name__)


def _dynamic_model_selector(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """ADK before_model_callback — fires before every LLM API call.

    Reads the user message, classifies the task, and mutates llm_request.model.
    The Gemini client uses llm_request.model directly in the API call.
    Returns None to tell ADK to proceed with the (now mutated) request.
    """
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if not task:
        logger.warning("before_model_callback: no user message found — keeping default model.")
        return None

    try:
        decision = classify_task(task, provider=settings.default_provider)
    except ClassificationError as exc:
        logger.error("Classification failed: %s — keeping default model.", exc)
        return None

    original = llm_request.model
    llm_request.model = decision.model_name

    logger.info(
        "Model selected | %s => %s [%s | %s | %s]",
        original,
        decision.model_name,
        decision.tier.value.upper(),
        decision.task_type.value,
        decision.complexity.value,
    )
    return None


root_agent = LlmAgent(
    name="DynamicModelAgent",
    model="gemini-2.5-flash",
    description="An agent that selects the right Gemini model per request based on task complexity.",
    instruction=(
        "You are a helpful expert assistant. "
        "Answer the user's question clearly and concisely."
    ),
    before_model_callback=_dynamic_model_selector,
)
