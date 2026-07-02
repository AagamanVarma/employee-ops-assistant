from app.services.query_classifier import classify_query


def test_workflow_queries_classify_as_workflow():
    result = classify_query("How do I apply leave?")
    assert result["intent"] == "workflow"
    assert result["confidence"] >= 0.6


def test_policy_queries_classify_as_policy():
    result = classify_query("How many sick leaves can interns take?")
    assert result["intent"] == "policy"
    assert result["confidence"] >= 0.6


def test_mixed_queries_classify_as_mixed():
    result = classify_query("Explain leave process and rules")
    assert result["intent"] == "mixed"


def test_unknown_queries_classify_as_unknown():
    result = classify_query("Who won IPL?")
    assert result["intent"] == "unknown"
