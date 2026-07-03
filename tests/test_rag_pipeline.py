from app.services.rag_pipeline import _build_grounding_context, _build_prompt


def test_grounding_context_is_structured_and_prompt_includes_context():
    chunks = [
        {
            "content": "Sick leave policy allows interns 3 sick leaves.",
            "source": "Employee Handbook",
            "citation": {"document": "Employee Handbook", "section": "Sick Leave Policy"},
        }
    ]
    workflows = [
        {
            "title": "Attendance Regularization",
            "description": "Regularize attendance in Zoho People",
            "steps": ["Open Zoho People", "Select attendance", "Submit regularization request"],
        }
    ]

    context_blob = _build_grounding_context(chunks, workflows)
    prompt = _build_prompt("How do I regularize attendance?", context_blob)

    assert "POLICY CONTEXT" in context_blob
    assert "WORKFLOW CONTEXT" in context_blob
    assert "Attendance Regularization" in context_blob
    assert "Context:" in prompt
    assert "How do I regularize attendance?" in prompt
