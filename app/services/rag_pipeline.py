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


def _build_grounding_context(chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        citation = chunk.get("citation") or {}
        source = citation.get("document") or chunk.get("source") or "Unknown document"
        page = citation.get("page")
        header = f"Policy Evidence {index}: {source}"
        if page:
            header = f"{header} (Page {page})"
        parts.append(f"{header}\n{chunk.get('content', '').strip()}")

    for index, workflow in enumerate(workflows, start=1):
        title = workflow.get("title") or "Workflow"
        description = workflow.get("description") or ""
        steps = workflow.get("steps") or []
        steps_text = "\n".join(f"- {step}" for step in steps)
        parts.append(
            f"Workflow Evidence {index}: {title}\nDescription: {description}\nSteps:\n{steps_text}"
        )

    return "\n\n".join(parts)


def _build_prompt(query: str) -> str:
    return (
        "You are a professional enterprise assistant.\n"
        "Use only the retrieved documents and workflows as context for your answer.\n"
        "Do not invent policies, workflow steps, or information not present in the retrieved context.\n"
        "For workflow-based queries, provide clear step-by-step guidance in natural language.\n"
        "For policy-based queries, summarize the relevant policy points clearly and professionally.\n"
        "If both documents and workflows are available, combine them into a coherent evidence-based answer.\n"
        "Do not use overly technical wording like 'grounded summary'.\n"
        "If you cannot answer from the retrieved context, write exactly: 'I do not have enough relevant context to answer this confidently.'\n\n"
        f"Employee question: {query}\n\n"
        "Context: {query}\n\n"
        "Answer in plain text, with a professional enterprise tone.\n"
        "If the answer is workflow-based, prefer a numbered list of steps.\n"
        "If the answer is policy-based, keep it concise and precise.\n"
        "If you cannot answer confidently from the retrieved context, write: 'I do not have enough relevant context to answer this confidently.'"
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


def _build_chunk_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    top_chunks = chunks[:2]
    sources = {chunk.get("citation", {}).get("document") or chunk.get("source") or "Document" for chunk in top_chunks}
    lines = ["Based on the retrieved policy content, here is the guidance:"]
    for chunk in top_chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        preview = content.replace("\n", " ").strip()
        if len(preview) > 300:
            preview = preview[:300].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {preview}")
    lines.append(f"\nSource: {', '.join(sorted(sources))}")
    return "\n".join(lines)


def _build_combined_answer(chunks: list[dict[str, Any]], workflows: list[dict[str, Any]]) -> str:
    workflow_answer = _build_workflow_answer(workflows)
    if not chunks:
        return workflow_answer
    if not workflows:
        return _build_chunk_answer(chunks)
    sources = {chunk.get("citation", {}).get("document") or chunk.get("source") or "Document" for chunk in chunks[:2]}
    lines = [workflow_answer, "\nSupported by policy context from:", ", ".join(sorted(sources))]
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
    prompt = _build_prompt(query)

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
