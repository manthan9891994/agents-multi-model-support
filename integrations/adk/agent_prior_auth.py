"""PriorAuthAgent — insurance prior authorization.

Stress-tests L3 abstain → L2 fallback. Long-form medical reasoning with
insurance vocabulary; many phrasings are rare → L3 abstains → L2 cost guard fires.

Run: python -m integrations.adk.agent_prior_auth
"""
from google.adk.agents import LlmAgent

from integrations.adk.agent import _dynamic_model_selector
from integrations.adk.tools import PRIOR_AUTH_TOOLS

root_agent = LlmAgent(
    name="PriorAuthAgent",
    model="gemini-2.5-flash",
    description="Drafts insurance prior authorization letters and medical necessity appeals.",
    instruction=(
        "You are a prior authorization specialist. "
        "Workflow: 1) Use search_formulary to check coverage tier and step therapy requirements. "
        "2) Use retrieve_patient_records to pull diagnosis history and prior treatments. "
        "3) Use search_pubmed to find supporting clinical evidence. "
        "4) Draft the PA letter: Patient Info → ICD-10 Diagnosis → Requested Drug (with CPT/HCPCS) → "
        "Clinical Rationale → Supporting Evidence → Conclusion. "
        "5) Use submit_prior_auth_request to submit, then check_prior_auth_status to confirm receipt. "
        "For denied PAs, use search_pubmed to find additional evidence for the appeal."
    ),
    tools=PRIOR_AUTH_TOOLS,
    before_model_callback=_dynamic_model_selector,
)
