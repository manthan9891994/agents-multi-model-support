"""ClinicalNoteAgent — clinical documentation generator.

PII detection MUST trigger on patient data → forces MEDIUM minimum even if
classifier said LOW. Tests compliance_flag + redaction.

Run: python -m integrations.adk.agent_clinical_note
"""
from google.adk.agents import LlmAgent

from integrations.adk.agent import _dynamic_model_selector
from integrations.adk.tools import CLINICAL_NOTE_TOOLS

root_agent = LlmAgent(
    name="ClinicalNoteAgent",
    model="gemini-2.5-flash",
    description="Generates admission notes, discharge summaries, SOAP notes, and progress notes.",
    instruction=(
        "You are a clinical documentation assistant. "
        "Workflow: 1) Use retrieve_previous_notes to check prior documentation before generating new notes. "
        "2) Use retrieve_patient_records when a patient ID is provided (triggers PII handling). "
        "3) Use search_icd10_codes to find accurate ICD-10 codes for all diagnoses. "
        "4) Use search_cpt_codes for procedure coding. "
        "5) Use search_clinical_guidelines to ensure assessment/plan aligns with current standards. "
        "Note structure: Subjective → Objective → Assessment (with ICD-10 codes) → Plan (with CPT codes). "
        "Never invent vitals, labs, or findings — use only what the user or tools provide."
    ),
    tools=CLINICAL_NOTE_TOOLS,
    before_model_callback=_dynamic_model_selector,
)
