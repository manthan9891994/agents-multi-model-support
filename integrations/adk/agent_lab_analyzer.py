"""LabAnalyzerAgent — lab result interpretation.

Diverse task_type coverage: ANALYZING, MATH, MULTIMODAL. Tests has_multimodal
context-signal path (radiology PDFs force tier bump) and tool-aware routing.

Run: python -m integrations.adk.agent_lab_analyzer
"""
from google.adk.agents import LlmAgent

from integrations.adk.agent import _dynamic_model_selector
from integrations.adk.tools import LAB_ANALYZER_TOOLS

root_agent = LlmAgent(
    name="LabAnalyzerAgent",
    model="gemini-2.5-flash",
    description="Interprets lab panels, calculates clinical scores, and flags anomalies.",
    instruction=(
        "You are a clinical lab interpretation assistant. "
        "Workflow: 1) Use get_lab_reference_ranges for every abnormal value — always state units and normal range. "
        "2) Use calculate_clinical_score when applicable (eGFR from creatinine, MELD for liver disease, "
        "HOMA-IR for insulin resistance, CHA2DS2-VASc for AF, CURB-65 for pneumonia severity). "
        "3) Use retrieve_patient_records only if a patient ID is provided. "
        "4) Use search_clinical_guidelines to connect lab findings to treatment recommendations. "
        "For trends: describe direction, magnitude, and likely etiology. "
        "Always flag critical values immediately."
    ),
    tools=LAB_ANALYZER_TOOLS,
    before_model_callback=_dynamic_model_selector,
)
