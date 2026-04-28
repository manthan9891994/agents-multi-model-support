"""ClinicalQAAgent — patient/clinician Q&A.

Widest tier spread per session: from "Hi" (CONVERSATION/SIMPLE → LOW) to
"How does ACE inhibitor renal dosing change with CKD stage 3?"
(REASONING/COMPLEX → HIGH). Best agent for measuring L3 intercept rate.

Run: python -m integrations.adk.agent_clinical_qa
"""
from google.adk.agents import LlmAgent

from integrations.adk.agent import _dynamic_model_selector
from integrations.adk.tools import CLINICAL_QA_TOOLS

root_agent = LlmAgent(
    name="ClinicalQAAgent",
    model="gemini-2.5-flash",
    description="Answers clinical questions for patients and clinicians.",
    instruction=(
        "You are a clinical Q&A assistant. Provide accurate, evidence-based answers. "
        "For clinician-facing questions, use search_clinical_guidelines to cite AHA/ACC, ADA, or KDIGO recommendations. "
        "Use search_drug_interactions when drug combinations are mentioned. "
        "Use calculate_clinical_score for any scoring (eGFR, MELD, HOMA-IR, CHA2DS2-VASc). "
        "Use get_lab_reference_ranges to interpret any lab values. "
        "Use retrieve_patient_records only when a patient ID is provided. "
        "For patient-facing questions, use plain language and recommend consulting their provider."
    ),
    tools=CLINICAL_QA_TOOLS,
    before_model_callback=_dynamic_model_selector,
)
