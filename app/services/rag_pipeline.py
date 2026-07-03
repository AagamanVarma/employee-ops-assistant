"""Retrieval-first grounded response pipeline.

This module orchestrates retrieval confidence checks and optional LLM-based
grounded summarization. It never allows weak retrieval context to be sent to
the LLM.
"""
from __future__ import annotations

import os
import re
from typing import Any

from app.llm import _get_api_key, generate_response
from app.services.retrieval import retrieve


FALLBACK_MESSAGE = (
    "I could not find a confident match for this query in the uploaded company policies. "
    "Please contact HR for clarification."
)


def _build_citations(chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        citation = chunk.get("citation") or {}
        source = citation.get("document") or chunk.get("source") or "Unknown document"
        page = citation.get("page")
        section = citation.get("section")
        label = source
        if page:
            label = f"{label} (Page {page})"
        if section:
            label = f"{label} - {section}"
        if label not in seen:
            citations.append(label)
            seen.add(label)

    for workflow in workflows:
        citation = workflow.get("citation") or {}
        workflow_name = citation.get("workflow") or workflow.get("title") or "Workflow"
        label = f"{workflow_name} Workflow"
        if label not in seen:
            citations.append(label)
            seen.add(label)

    return citations


def _dedupe_context_items(items: list[dict[str, Any]], key_field: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in items:
        key = (item.get(key_field) or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def _clean_text(text: str, *, max_chars: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def _build_policy_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    unique_chunks = _dedupe_context_items(chunks, "content")
    sections: list[str] = []
    for index, chunk in enumerate(unique_chunks[:3], start=1):
        citation = chunk.get("citation") or {}
        source = citation.get("document") or chunk.get("source") or "Unknown document"
        section = citation.get("section") or chunk.get("section_title") or ""
        content = _clean_text(chunk.get("content") or "")
        line = f"{index}. {source}"
        if section:
            line = f"{line} | Section: {section}"
        sections.append(f"{line}\n   - {content}")
    return "POLICY CONTEXT\n" + "\n".join(sections)


def _build_workflow_context(workflows: list[dict[str, Any]]) -> str:
    if not workflows:
        return ""
    unique_workflows = _dedupe_context_items(workflows, "title")
    sections: list[str] = []
    for index, workflow in enumerate(unique_workflows[:2], start=1):
        title = workflow.get("title") or "Workflow"
        description = _clean_text(workflow.get("description") or "")
        steps = [step for step in workflow.get("steps") or [] if (step or "").strip()]
        steps_text = "\n".join(f"   {step_index}. {step}" for step_index, step in enumerate(steps[:5], start=1))
        section_lines = [f"{index}. Workflow: {title}"]
        if description:
            section_lines.append(f"   Description: {description}")
        if steps_text:
            section_lines.append("   Steps:")
            section_lines.append(steps_text)
        sections.append("\n".join(section_lines))
    return "WORKFLOW CONTEXT\n" + "\n".join(sections)


def _build_grounding_context(chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> str:
    policy_context = _build_policy_context(chunks)
    workflow_context = _build_workflow_context(workflows)
    parts: list[str] = []
    if policy_context:
        parts.append(policy_context)
    if workflow_context:
        parts.append(workflow_context)
    if not parts:
        return ""
    return "\n\n".join(parts)


def _build_prompt(query: str, context_blob: str) -> str:
    return (
        "You are a professional enterprise assistant.\n"
        "Answer only from the provided context.\n"
        "Do not invent policies, workflow steps, or facts.\n"
        "If the provided context is insufficient, say that you need HR clarification.\n"
        "For operational queries involving apply, submit, request, regularize, report, or complete a process, prioritize workflow guidance.\n"
        "When workflow context is provided, present the steps in the same order as given and use numbered steps.\n"
        "For informational queries about rules, eligibility, limits, or benefits, prefer concise policy explanation.\n"
        "When both policy and workflow context are provided, answer with the workflow first and then add brief policy support.\n"
        "Use a professional, employee-friendly tone.\n"
        "Keep the response structured and easy to scan.\n"
        "If you cannot answer confidently from the context, write exactly: 'I do not have enough relevant context to answer this confidently.'\n\n"
        f"Employee question: {query}\n\n"
        f"Context:\n{context_blob}\n\n"
        "Answer in plain text with short paragraphs or numbered steps where appropriate."
    )


def _build_workflow_answer(workflows: list[dict[str, Any]]) -> str:
    if not workflows:
        return ""
    workflow = workflows[0]
    title = workflow.get("title") or "Workflow"
    steps = workflow.get("steps") or []
    if not steps:
        description = workflow.get("description") or "Follow the workflow steps as provided."
        return f"Based on the {title} workflow:\n{description}\n\nSource: {title} Workflow"

    lines = [f"Based on the {title} workflow, follow these steps:"]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step}")
    lines.append(f"\nSource: {title} Workflow")
    return "\n".join(lines)


def _format_policy_support(chunks: list[dict[str, Any]]) -> str:
    top_chunks = chunks[:2]
    sources = {chunk.get("citation", {}).get("document") or chunk.get("source") or "Document" for chunk in top_chunks}
    lines = ["Policy support:"]
    for chunk in top_chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        preview = content.replace("\n", " ").strip()
        if len(preview) > 260:
            preview = preview[:260].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {preview}")
    if sources:
        lines.append(f"Source: {', '.join(sorted(sources))}")
    return "\n".join(lines)


def _build_chunk_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    lines = ["Based on the retrieved policy content, here is the guidance:"]
    policy_support = _format_policy_support(chunks)
    if policy_support:
        lines.append(policy_support)
    return "\n".join(lines)


def _build_combined_answer(chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> str:
    workflow_answer = _build_workflow_answer(workflows)
    if not chunks:
        return workflow_answer
    if not workflows:
        return _build_chunk_answer(chunks)
    lines = [workflow_answer, "", "Supporting policy context:"]
    support_text = _format_policy_support(chunks)
    if support_text:
        lines.append(support_text)
    return "\n".join(lines)


def _is_llm_fallback_response(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    # Accept either legacy or updated fallback phrasing from the LLM
    return normalized in (
        "i do not have enough grounded policy context to answer this confidently.",
        "i do not have enough relevant context to answer this confidently.",
    )


def _extractive_summary(query: str, chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> str:
    summary_lines = [f"Summary for: {query}"]
    if chunks:
        summary_lines.append("Policy highlights:")
        for chunk in chunks[:2]:
            content = (chunk.get("content") or "").strip()
            if content:
                preview = content[:500]
                if len(content) > 500:
                    preview = preview.rstrip() + "..."
                summary_lines.append(f"- {preview}")
    if workflows:
        summary_lines.append("Related workflow guidance:")
        for workflow in workflows[:2]:
            title = workflow.get("title") or "Workflow"
            steps = workflow.get("steps") or []
            if steps:
                summary_lines.append(f"- {title}: {steps[0]}")
            else:
                summary_lines.append(f"- {title}")
    return "\n".join(summary_lines)


def run_rag_pipeline(query: str, top_k: int = 5) -> dict[str, Any]:
    retrieval_result = retrieve(query, top_k=top_k)
    confidence = retrieval_result.get("confidence", {})
    chunks = retrieval_result.get("chunks", [])
    workflows = retrieval_result.get("workflows", [])
    debug = retrieval_result.get("debug", {})

    has_retrieval_context = bool(chunks or workflows)
    is_confident = bool(confidence.get("is_confident", False))
    if not is_confident and not has_retrieval_context:
        confidence["fallback_message"] = FALLBACK_MESSAGE
        return {
            "query": query,
            "answer": None,
            "answer_source": "fallback",
            "chunks": chunks,
            "workflows": workflows,
            "citations": [],
            "confidence": confidence,
            "debug": {
                **debug,
                "fallback_triggered": True,
                "retrieved_chunk_count": len(chunks),
                "retrieved_workflow_count": len(workflows),
                "llm_called": False,
                "llm_error": None,
            },
        }

    context_blob = _build_grounding_context(chunks, workflows)
    prompt = _build_prompt(query, context_blob)

    llm_enabled = bool(_get_api_key())
    llm_called = False
    llm_output = None
    llm_error = None
    llm_source = "fallback"
    if llm_enabled and context_blob.strip():
        llm_called = True
        llm_resp = generate_response(
            prompt=prompt,
            context=[{"text": context_blob}],
            model=os.getenv("GEN_MODEL", "gemini-2.5-flash-lite"),
        )
        llm_output = (llm_resp or {}).get("text")
        llm_error = (llm_resp or {}).get("error")
        llm_source = (llm_resp or {}).get("source") or "fallback"

    answer_text = (llm_output or "").strip()

    # If the LLM explicitly refused with a fallback phrase or returned empty,
    # attempt one retry asking it to synthesize strictly from the provided context.
    if _is_llm_fallback_response(answer_text) or not answer_text:
        if llm_enabled and context_blob.strip():
            forced_prompt = (
                "Using ONLY the provided context, produce a concise, professional answer to the employee question. "
                "Do not state that you lack context unless it is truly impossible. Use the context to synthesize a grounded response.\n\n"
                + prompt
            )
            retry_resp = generate_response(
                prompt=forced_prompt,
                context=[{"text": context_blob}],
                model=os.getenv("GEN_MODEL", "gemini-2.5-flash-lite"),
            )
            retry_text = (retry_resp or {}).get("text") or ""
            retry_source = (retry_resp or {}).get("source") or llm_source
            if not _is_llm_fallback_response(retry_text) and retry_text.strip():
                answer_text = retry_text.strip()
                llm_source = retry_source

        # If after retry there's still no LLM answer, return a constructed combined answer
        if not answer_text:
            if workflows or chunks:
                answer_text = _build_combined_answer(chunks, workflows)
                llm_source = "combined"
            else:
                confidence["fallback_message"] = FALLBACK_MESSAGE
                return {
                    "query": query,
                    "answer": None,
                    "answer_source": "fallback",
                    "chunks": [],
                    "workflows": [],
                    "citations": [],
                    "confidence": confidence,
                    "debug": {
                        **debug,
                        "fallback_triggered": True,
                        "fallback_reason": "LLM did not return a polished answer and no grounding context was available.",
                        "retrieved_chunk_count": len(chunks),
                        "retrieved_workflow_count": len(workflows),
                        "llm_called": llm_called,
                        "llm_source": llm_source,
                        "llm_error": llm_error,
                    },
                }

    citations = _build_citations(chunks, workflows)

    return {
        "query": query,
        "answer": answer_text,
        "answer_source": llm_source,
        "chunks": chunks,
        "workflows": workflows,
        "citations": citations,
        "confidence": confidence,
        "debug": {
            **debug,
            "fallback_triggered": False,
            "retrieved_chunk_count": len(chunks),
            "retrieved_workflow_count": len(workflows),
            "llm_called": llm_called,
            "llm_source": llm_source,
            "llm_error": llm_error,
        },
    }
