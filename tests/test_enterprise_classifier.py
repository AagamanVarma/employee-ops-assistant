from app.services.query_classifier import classify_query


def test_fde_role_query_is_not_unknown():
    result = classify_query("Tell me about FDE role")
    assert result["intent"] in {"policy", "mixed"}
    assert result["matched_enterprise_terms"]


def test_pip_process_query_is_not_unknown():
    result = classify_query("What is PIP process?")
    assert result["intent"] in {"workflow", "mixed"}
    assert result["matched_enterprise_terms"]


def test_lop_query_is_policy():
    result = classify_query("What is LOP?")
    assert result["intent"] == "policy"


def test_wfh_policy_query_is_policy():
    result = classify_query("What is WFH policy?")
    assert result["intent"] == "policy"
