"""Dummy healthcare tools for ADK agent testing.

Covers every routing scenario:
  - RAG / vector search          → raises available_tools count → tier bump
  - Clinical computation         → MATH task_type path
  - PII-bearing retrieval        → compliance_flag → MEDIUM floor
  - External API simulation      → doc_creation / code_creation paths
  - Long-running async ops       → call_number > 1 path
  - Error injection              → has_error context signal path

All functions return realistic-looking dummy data — no real API calls.
"""
from __future__ import annotations

import random
import time
from typing import Optional


# ── RAG / Knowledge Retrieval ─────────────────────────────────────────────────

def search_clinical_guidelines(
    query: str,
    guideline_source: str = "AHA/ACC",
    max_results: int = 3,
) -> dict:
    """Search clinical practice guidelines using semantic vector search.

    Args:
        query: Clinical question or topic to search for.
        guideline_source: Guideline body to search (AHA/ACC, ADA, KDIGO, IDSA, NICE).
        max_results: Maximum number of guideline excerpts to return.

    Returns:
        Dictionary with matched guideline excerpts, recommendation class, and evidence level.
    """
    sources = {
        "AHA/ACC": [
            "2023 ACC/AHA Guideline for Diagnosis and Management of Heart Failure",
            "2022 AHA/ACC/HFSA Guideline for Heart Failure",
            "2021 ACC/AHA Guideline on Chest Pain",
        ],
        "ADA": [
            "ADA Standards of Care in Diabetes — 2024",
            "ADA Pharmacologic Approaches to Glycemic Treatment 2024",
        ],
        "KDIGO": [
            "KDIGO 2024 CKD Guideline",
            "KDIGO AKI Guideline 2012 (updated 2023)",
        ],
    }
    docs = sources.get(guideline_source, sources["AHA/ACC"])
    return {
        "query": query,
        "source": guideline_source,
        "results": [
            {
                "document": random.choice(docs),
                "section": f"Section {random.randint(3, 9)}.{random.randint(1, 6)}",
                "excerpt": f"[Dummy excerpt for '{query}'] Recommendation class I, Evidence level A.",
                "recommendation_class": random.choice(["I", "IIa", "IIb", "III"]),
                "evidence_level": random.choice(["A", "B-R", "B-NR", "C-LD"]),
            }
            for _ in range(min(max_results, len(docs)))
        ],
    }


def search_drug_interactions(drug_a: str, drug_b: str) -> dict:
    """Check for clinically significant drug-drug interactions using a drug database.

    Args:
        drug_a: First drug name (generic or brand).
        drug_b: Second drug name (generic or brand).

    Returns:
        Interaction severity, mechanism, clinical effect, and management recommendation.
    """
    severity = random.choice(["none", "minor", "moderate", "major", "contraindicated"])
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "severity": severity,
        "mechanism": "[Dummy] CYP3A4 inhibition increases plasma concentration." if severity != "none" else "No known interaction.",
        "clinical_effect": "[Dummy] Risk of QT prolongation." if severity in ("major", "contraindicated") else "Minimal clinical impact.",
        "management": "Avoid combination." if severity == "contraindicated" else "Monitor closely.",
    }


def retrieve_patient_records(
    patient_id: str,
    sections: Optional[list[str]] = None,
) -> dict:
    """Retrieve patient medical records from the EHR system. Contains PII.

    Args:
        patient_id: Unique patient identifier (MRN).
        sections: Specific record sections to retrieve (medications, allergies, labs, notes).

    Returns:
        Patient demographics and requested clinical sections. Contains PII — triggers compliance flag.
    """
    # This tool returns PII → forces MEDIUM tier minimum via compliance_flag
    sections = sections or ["demographics", "medications", "allergies"]
    return {
        "patient_id": patient_id,
        "name": "John Doe",           # dummy PII
        "dob": "1958-03-14",
        "mrn": patient_id,
        "allergies": ["Penicillin (anaphylaxis)", "Sulfa drugs (rash)"],
        "active_medications": [
            {"drug": "Metformin", "dose": "1000 mg", "frequency": "BID"},
            {"drug": "Lisinopril", "dose": "10 mg",  "frequency": "daily"},
            {"drug": "Atorvastatin", "dose": "40 mg", "frequency": "nightly"},
        ],
        "sections_returned": sections,
        "note": "[DUMMY DATA — no real patient data]",
    }


def search_pubmed(query: str, max_results: int = 5, min_year: int = 2020) -> dict:
    """Search PubMed for clinical evidence supporting a treatment or intervention.

    Args:
        query: Clinical question or PICO-formatted search string.
        max_results: Maximum number of abstracts to return.
        min_year: Only include studies published after this year.

    Returns:
        List of relevant PubMed abstracts with PMID, title, and summary.
    """
    return {
        "query": query,
        "total_found": random.randint(12, 340),
        "results": [
            {
                "pmid": f"3{random.randint(1000000, 9999999)}",
                "title": f"[Dummy] Randomized Controlled Trial of Treatment for {query[:40]}",
                "journal": random.choice(["NEJM", "JAMA", "Lancet", "JACC", "Diabetes Care"]),
                "year": random.randint(min_year, 2025),
                "study_type": random.choice(["RCT", "Meta-analysis", "Cohort study", "Case-control"]),
                "abstract_summary": f"[Dummy abstract] This study evaluated {query[:50]}... Results showed significant improvement (p<0.05).",
                "n_patients": random.randint(200, 12000),
            }
            for _ in range(min(max_results, 5))
        ],
    }


# ── Clinical Computation ──────────────────────────────────────────────────────

def calculate_clinical_score(
    score_name: str,
    parameters: dict,
) -> dict:
    """Calculate a validated clinical scoring tool from lab values and patient parameters.

    Args:
        score_name: Name of scoring system (eGFR, MELD, CHA2DS2-VASc, HOMA-IR, Child-Pugh, CURB-65, HbA1c-to-avg-glucose).
        parameters: Dictionary of required input values for the score.

    Returns:
        Calculated score, interpretation, risk category, and recommended action.
    """
    score_val = round(random.uniform(3.0, 95.0), 1)
    interpretations = {
        "eGFR":           f"CKD Stage {random.choice(['2', '3a', '3b', '4'])} — {score_val:.0f} mL/min/1.73m²",
        "MELD":           f"MELD score {score_val:.0f} — {'High' if score_val > 20 else 'Moderate'} 90-day mortality risk",
        "CHA2DS2-VASc":   f"Score {score_val:.0f} — {'Anticoagulation recommended' if score_val >= 2 else 'Low risk'}",
        "HOMA-IR":        f"HOMA-IR {score_val:.1f} — {'Insulin resistance present' if score_val > 2.5 else 'Normal'}",
        "CURB-65":        f"CURB-65 score {score_val:.0f} — {'Hospital admission recommended' if score_val >= 2 else 'Outpatient treatment'}",
    }
    return {
        "score_name": score_name,
        "inputs": parameters,
        "score": score_val,
        "interpretation": interpretations.get(score_name, f"[Dummy] {score_name} = {score_val}"),
        "risk_category": random.choice(["low", "moderate", "high", "very high"]),
        "guideline_recommendation": f"[Dummy] Per current guidelines, this score warrants {random.choice(['monitoring', 'intervention', 'specialist referral'])}.",
    }


def get_lab_reference_ranges(test_name: str, patient_age: int = 50, sex: str = "M") -> dict:
    """Retrieve normal reference ranges for a laboratory test adjusted for age and sex.

    Args:
        test_name: Name of the lab test (e.g., Hemoglobin, Creatinine, TSH, ALT).
        patient_age: Patient age in years for age-adjusted ranges.
        sex: Biological sex for sex-specific ranges (M or F).

    Returns:
        Reference range, units, critical values, and interpretation guidance.
    """
    ranges = {
        "Hemoglobin":  {"low": 13.5 if sex == "M" else 12.0, "high": 17.5 if sex == "M" else 15.5, "unit": "g/dL"},
        "Creatinine":  {"low": 0.74, "high": 1.35 if sex == "M" else 1.04, "unit": "mg/dL"},
        "ALT":         {"low": 7, "high": 56, "unit": "U/L"},
        "AST":         {"low": 10, "high": 40, "unit": "U/L"},
        "TSH":         {"low": 0.4, "high": 4.0, "unit": "mIU/L"},
        "HbA1c":       {"low": 4.0, "high": 5.6, "unit": "%"},
        "BUN":         {"low": 7, "high": 20, "unit": "mg/dL"},
        "Sodium":      {"low": 136, "high": 145, "unit": "mEq/L"},
        "Potassium":   {"low": 3.5, "high": 5.0, "unit": "mEq/L"},
    }
    ref = ranges.get(test_name, {"low": 0, "high": 100, "unit": "units"})
    return {
        "test": test_name,
        "age": patient_age,
        "sex": sex,
        "normal_low":  ref["low"],
        "normal_high": ref["high"],
        "unit": ref["unit"],
        "critical_low":  ref["low"] * 0.6,
        "critical_high": ref["high"] * 1.5,
        "note": "Age-adjusted ranges applied for patients >65." if patient_age > 65 else None,
    }


# ── Insurance / Prior Authorization ──────────────────────────────────────────

def search_formulary(
    drug_name: str,
    insurance_plan: str = "BCBS PPO",
    diagnosis_code: Optional[str] = None,
) -> dict:
    """Search insurance formulary for drug coverage tier, step therapy, and PA requirements.

    Args:
        drug_name: Generic or brand drug name.
        insurance_plan: Insurance plan identifier.
        diagnosis_code: ICD-10 diagnosis code for indication-specific coverage lookup.

    Returns:
        Formulary tier, PA required flag, step therapy requirements, and quantity limits.
    """
    tier = random.randint(1, 5)
    pa_required = tier >= 3 or drug_name.lower() in ("humira", "adalimumab", "infliximab", "remicade")
    return {
        "drug": drug_name,
        "plan": insurance_plan,
        "formulary_tier": tier,
        "tier_label": ["Preferred Generic", "Generic", "Preferred Brand", "Non-Preferred Brand", "Specialty"][tier - 1],
        "pa_required": pa_required,
        "step_therapy_required": pa_required,
        "step_therapy_drugs": ["Methotrexate (8 weeks)", "Leflunomide (8 weeks)"] if pa_required else [],
        "quantity_limit": "30 injections/90 days" if pa_required else None,
        "copay_tier": f"${[10, 40, 80, 120, 250][tier - 1]}/month",
        "diagnosis_code": diagnosis_code,
    }


def submit_prior_auth_request(
    patient_id: str,
    drug_name: str,
    diagnosis_code: str,
    clinical_justification: str,
    prescriber_npi: str,
) -> dict:
    """Submit a prior authorization request to the insurance payer system.

    Args:
        patient_id: Patient MRN or insurance member ID.
        drug_name: Drug requiring prior authorization.
        diagnosis_code: Primary ICD-10 diagnosis code.
        clinical_justification: Clinical narrative supporting medical necessity.
        prescriber_npi: Prescribing physician NPI number.

    Returns:
        PA submission confirmation with tracking number and expected decision timeline.
    """
    time.sleep(0.1)  # simulate network call
    return {
        "status": "submitted",
        "tracking_number": f"PA-{random.randint(100000, 999999)}",
        "drug": drug_name,
        "patient_id": patient_id,
        "payer_received_at": "2026-04-27T18:00:00Z",
        "expected_decision_by": "2026-04-29T17:00:00Z",
        "expedited_available": True,
        "next_steps": "Payer will contact prescriber within 2 business days. Expedited review available for urgent cases.",
    }


def check_prior_auth_status(tracking_number: str) -> dict:
    """Check the current status of a submitted prior authorization request.

    Args:
        tracking_number: PA tracking number returned at submission.

    Returns:
        Current PA status, decision details, and any required additional information.
    """
    status = random.choice(["pending", "approved", "denied", "more_info_needed"])
    return {
        "tracking_number": tracking_number,
        "status": status,
        "decision": "Approved for 12 months" if status == "approved" else None,
        "denial_reason": "Step therapy not completed" if status == "denied" else None,
        "info_needed": ["Recent labs within 90 days", "Specialist consult note"] if status == "more_info_needed" else [],
        "appeal_deadline": "2026-05-27" if status == "denied" else None,
    }


# ── Documentation / Coding ────────────────────────────────────────────────────

def search_icd10_codes(description: str, max_results: int = 5) -> dict:
    """Search ICD-10-CM codes by clinical description using fuzzy matching.

    Args:
        description: Clinical condition description to search.
        max_results: Maximum number of matching codes to return.

    Returns:
        List of matching ICD-10 codes with descriptions and hierarchical path.
    """
    mock_codes = {
        "diabetes": [("E11.9", "Type 2 diabetes mellitus without complications"), ("E11.65", "T2DM with hyperglycemia"), ("E11.40", "T2DM with diabetic neuropathy")],
        "hypertension": [("I10", "Essential (primary) hypertension"), ("I13.10", "HTN with CKD stage 1-4")],
        "ckd": [("N18.3", "Chronic kidney disease, stage 3"), ("N18.31", "CKD stage 3a"), ("N18.32", "CKD stage 3b")],
        "pneumonia": [("J18.9", "Pneumonia, unspecified"), ("J15.9", "Unspecified bacterial pneumonia")],
        "copd": [("J44.1", "COPD with acute exacerbation"), ("J44.0", "COPD with acute lower respiratory infection")],
        "aki": [("N17.9", "Acute kidney failure, unspecified"), ("N17.0", "AKI with tubular necrosis")],
    }
    key = next((k for k in mock_codes if k in description.lower()), None)
    codes = mock_codes.get(key, [("Z99.89", "Dependence on other enabling machines and devices")])
    return {
        "query": description,
        "results": [{"code": c, "description": d, "billable": True} for c, d in codes[:max_results]],
    }


def search_cpt_codes(procedure_description: str) -> dict:
    """Search CPT procedure codes by clinical description.

    Args:
        procedure_description: Description of the medical procedure or service.

    Returns:
        Matching CPT codes with RVU values and documentation requirements.
    """
    mock_cpts = [
        {"code": "99213", "description": "Office visit, established patient, moderate complexity", "rvu": 1.3},
        {"code": "99215", "description": "Office visit, established patient, high complexity", "rvu": 2.11},
        {"code": "93000", "description": "Electrocardiogram with interpretation", "rvu": 0.17},
        {"code": "71046", "description": "Chest X-ray, 2 views", "rvu": 0.22},
        {"code": "80053", "description": "Comprehensive metabolic panel", "rvu": 0.0},
    ]
    return {
        "query": procedure_description,
        "results": random.sample(mock_cpts, min(3, len(mock_cpts))),
        "note": "[Dummy] Verify coding with certified coder before submission.",
    }


def retrieve_previous_notes(patient_id: str, note_type: str = "progress", limit: int = 3) -> dict:
    """Retrieve previous clinical notes for a patient from the EHR. Contains PII.

    Args:
        patient_id: Patient MRN.
        note_type: Type of note to retrieve (progress, discharge, consult, operative).
        limit: Maximum number of notes to retrieve.

    Returns:
        List of previous clinical notes with date, author, and content. Contains PII.
    """
    return {
        "patient_id": patient_id,
        "note_type": note_type,
        "notes": [
            {
                "date": f"2026-0{random.randint(1, 4)}-{random.randint(10, 28)}",
                "author": f"Dr. {random.choice(['Smith', 'Patel', 'Johnson'])}",
                "specialty": random.choice(["Internal Medicine", "Cardiology", "Nephrology"]),
                "content": f"[Dummy {note_type} note] Patient presented with... Assessment: stable. Plan: continue current medications.",
            }
            for _ in range(limit)
        ],
        "note": "[DUMMY DATA — no real patient data]",
    }


# ── Tool collections per agent ────────────────────────────────────────────────

CLINICAL_QA_TOOLS = [
    search_clinical_guidelines,
    search_drug_interactions,
    retrieve_patient_records,
    search_pubmed,
    calculate_clinical_score,
    get_lab_reference_ranges,
]

PRIOR_AUTH_TOOLS = [
    search_formulary,
    submit_prior_auth_request,
    check_prior_auth_status,
    search_pubmed,
    retrieve_patient_records,
]

LAB_ANALYZER_TOOLS = [
    get_lab_reference_ranges,
    calculate_clinical_score,
    retrieve_patient_records,
    search_clinical_guidelines,
]

CLINICAL_NOTE_TOOLS = [
    search_icd10_codes,
    search_cpt_codes,
    retrieve_previous_notes,
    retrieve_patient_records,
    search_clinical_guidelines,
]
