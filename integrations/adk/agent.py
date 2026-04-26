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


_call_counter: dict[str, int] = {}
_ERROR_SIGNALS = {"error", "exception", "traceback", "failed", "failure", "timeout", "refused"}


def _extract_context_signals(llm_request: LlmRequest, agent_name: str):
    from classifier.core.types import ContextSignals

    _call_counter[agent_name] = _call_counter.get(agent_name, 0) + 1
    call_number = _call_counter[agent_name]

    total_chars   = 0
    last_role     = "user"
    last_non_user = ""
    has_multimodal = False

    for content in llm_request.contents:
        last_role = content.role or "user"
        for part in (content.parts or []):
            text = getattr(part, "text", "") or ""
            total_chars += len(text)
            if content.role in ("tool", "model"):
                last_non_user = text
            # Item 3: Detect actual multimodal data parts (image/audio/file bytes)
            if (
                getattr(part, "inline_data", None) is not None
                or getattr(part, "file_data", None) is not None
            ):
                has_multimodal = True

    has_error = False
    if last_non_user:
        lower = last_non_user[-2000:].lower()
        has_error = any(sig in lower for sig in _ERROR_SIGNALS)

    # Item 4: Count tools available to the agent
    available_tools = len(getattr(llm_request, "tools", None) or [])

    return ContextSignals(
        total_context_tokens=total_chars // 4,
        call_number=call_number,
        has_error=has_error,
        last_role=last_role,
        has_multimodal=has_multimodal,
        available_tools=available_tools,
    )


def _dynamic_model_selector(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """ADK before_model_callback — fires before every LLM API call."""
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if not task:
        logger.warning("before_model_callback: no user message found — keeping default model.")
        return None

    ctx_signals = _extract_context_signals(llm_request, agent_name="DynamicModelAgent")

    try:
        decision = classify_task(
            task,
            provider=settings.default_provider,
            context_signals=ctx_signals,
        )
    except ClassificationError as exc:
        logger.error("Classification failed: %s — keeping default model.", exc)
        return None

    original = llm_request.model
    llm_request.model = decision.model_name

    logger.info(
        "Model selected | %s => %s [%s | %s | %s | call=%d | ctx_tokens=%d%s%s]",
        original,
        decision.model_name,
        decision.tier.value.upper(),
        decision.task_type.value,
        decision.complexity.value,
        ctx_signals.call_number,
        ctx_signals.total_context_tokens,
        " | PII" if decision.compliance_flag else "",
        f" | tools={ctx_signals.available_tools}" if ctx_signals.available_tools else "",
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
