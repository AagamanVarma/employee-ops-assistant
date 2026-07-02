"""Lightweight query intent classification for retrieval orchestration."""
from __future__ import annotations

import re
from typing import Any

WORKFLOW_KEYWORDS = (
    "how",
    "process",
    "steps",
    "step",
    "apply",
    "submit",
    "regularize",
    "fill",
    "create",
    "request",
    "approve",
    "approval",
    "raise",
    "update",
    "initiate",
    "follow",
)

POLICY_KEYWORDS = (
    "policy",
    "rule",
    "rules",
    "allowed",
    "allow",
    "eligibility",
    "eligible",
    "balance",
    "limit",
    "limits",
    "reimbursement",
    "reimburse",
    "quota",
    "entitled",
    "entitlement",
    "maternity",
    "sick",
    "lop",
    "benefit",
    "benefits",
    "leave",
)

ENTERPRISE_TERMS = {
    "fde": "role",
    "wfh": "policy",
    "pip": "process",
    "lop": "policy",
    "lta": "policy",
    "kra": "performance",
    "kpi": "performance",
    "sop": "process",
    "pto": "policy",
    "hr": "policy",
    "ops": "process",
    "role": "role",
    "process": "process",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def classify_query(query: str) -> dict[str, Any]:
    """Classify a user query as workflow, policy, mixed, or unknown.

    The logic is intentionally simple and deterministic: count keyword hits for
    workflow and policy intent, and use a small confidence score to choose the
    most likely label.
    """

    normalized = _normalize(query)
    if not normalized:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "empty query",
            "workflow_score": 0,
            "policy_score": 0,
        }

    workflow_score = 0
    policy_score = 0
    matched_terms: list[str] = []
    acronym_matches: list[str] = []

    if re.search(r"\bhow do i\b|\bhow to\b", normalized):
        workflow_score += 2
    if re.search(r"\bwhat is\b|\bwhat are\b|\bhow many\b", normalized):
        policy_score += 1

    for keyword in WORKFLOW_KEYWORDS:
        if keyword in normalized:
            workflow_score += 1

    for keyword in POLICY_KEYWORDS:
        if keyword in normalized:
            policy_score += 1

    acronym_tokens = re.findall(r"\b[a-z]{2,4}\b", normalized)
    for token in acronym_tokens:
        if token in ENTERPRISE_TERMS:
            matched_terms.append(token)
            acronym_matches.append(token)
            if ENTERPRISE_TERMS[token] == "process":
                workflow_score += 1
            elif ENTERPRISE_TERMS[token] == "policy":
                policy_score += 1
            elif ENTERPRISE_TERMS[token] == "role":
                policy_score += 1

    if "role" in normalized or "roles" in normalized:
        policy_score += 1

    if "process" in normalized and "policy" in normalized:
        workflow_score += 1
        policy_score += 1

    if "and" in normalized and workflow_score and policy_score:
        workflow_score += 0.5
        policy_score += 0.5

    has_workflow_signal = workflow_score >= 2
    has_policy_signal = policy_score >= 2
    workflow_cue = any(term in normalized for term in ("process", "steps", "step", "apply", "submit", "regularize", "fill", "create", "request", "approve", "approval", "raise", "update", "initiate", "follow"))
    policy_cue = any(term in normalized for term in ("policy", "rule", "rules", "allowed", "allow", "eligibility", "eligible", "balance", "limit", "limits", "reimbursement", "reimburse", "quota", "entitled", "entitlement", "maternity", "sick", "lop", "benefit", "benefits", "leave"))
    mentions_both = workflow_cue and policy_cue and (
        " and " in normalized
        or "process" in normalized
        or "steps" in normalized
        or "policy" in normalized
        or "rule" in normalized
        or "rules" in normalized
    )

    if mentions_both or (has_workflow_signal and has_policy_signal):
        intent = "mixed"
        confidence = min(0.95, 0.65 + 0.07 * max(workflow_score, policy_score))
        reason = "both workflow and policy signals were detected"
    elif workflow_score >= 2 and workflow_score > policy_score:
        intent = "workflow"
        confidence = min(0.95, 0.7 + 0.05 * workflow_score)
        reason = "workflow-oriented language was detected"
    elif policy_score >= 2 and policy_score > workflow_score:
        intent = "policy"
        confidence = min(0.95, 0.7 + 0.05 * policy_score)
        reason = "policy-oriented language was detected"
    elif workflow_score >= 2 and policy_score >= 2:
        intent = "mixed"
        confidence = min(0.95, 0.65 + 0.07 * max(workflow_score, policy_score))
        reason = "both workflow and policy signals were detected"
    elif matched_terms and (policy_score >= 1 or workflow_score >= 1):
        intent = "policy" if policy_score >= workflow_score else "mixed"
        confidence = 0.72
        reason = "enterprise terminology matched the query"
    else:
        intent = "unknown"
        confidence = 0.4
        reason = "no strong intent signals were detected"

    return {
        "intent": intent,
        "confidence": round(confidence, 2),
        "reason": reason,
        "workflow_score": int(workflow_score),
        "policy_score": int(policy_score),
        "matched_enterprise_terms": matched_terms,
        "acronym_matches": acronym_matches,
    }
