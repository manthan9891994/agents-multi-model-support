from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from classifier import classify_task


def _dynamic_model_selector(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    """
    Fires before every LLM call inside ADK's _call_llm_async().
    Classifies the user's task and mutates llm_request.model.
    Gemini client uses llm_request.model for the actual API call.
    Returning None tells ADK to proceed with the (mutated) request.
    """
    task = ""
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            task = content.parts[0].text or ""
            break

    if task:
        decision = classify_task(task, provider="google")
        original = llm_request.model
        llm_request.model = decision.model_name
        print(
            f"\n[classifier] task   : {task[:70]}"
            f"\n[classifier] model  : {original} => {decision.model_name}"
            f"\n[classifier] reason : {decision.task_type.value} / {decision.complexity.value} => {decision.tier.value.upper()}\n"
        )

    return None


# ONE agent. The model on it is just the fallback if task is empty.
# _dynamic_model_selector swaps it before every real API call.
root_agent = LlmAgent(
    name="DynamicModelAgent",
    model="gemini-2.5-flash",
    description="An agent that picks the right Gemini model based on task complexity.",
    instruction=(
        "You are a helpful expert assistant. "
        "Answer the user's question clearly and concisely."
    ),
    before_model_callback=_dynamic_model_selector,
)
