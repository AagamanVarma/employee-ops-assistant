"""Semantic retrieval that combines ChromaDB chunk search with simple workflow lookup.

This module provides `retrieve(query)` which returns top chunks and matching workflows.
"""
from app.services.embeddings import emb_service
from app import models
from app.database import SessionLocal
from sqlalchemy.orm import Session
import math


def _cosine_similarity(query_vec, candidate_vec):
    dot = sum(q * c for q, c in zip(query_vec, candidate_vec))
    q_norm = math.sqrt(sum(q * q for q in query_vec))
    c_norm = math.sqrt(sum(c * c for c in candidate_vec))
    if q_norm == 0 or c_norm == 0:
        return 0.0
    return dot / (q_norm * c_norm)


def _search_workflows(db: Session, query: str, limit: int = 5):
    q = f"%{query}%"
    results = db.query(models.Workflow).filter(
        (models.Workflow.name.ilike(q)) | (models.Workflow.description.ilike(q))
    ).limit(limit).all()
    return results


def _workflow_scores(query: str, workflows):
    if not workflows:
        return []
    texts = [f"{wf.name} {wf.description or ''}" for wf in workflows]
    query_emb = emb_service.embed_texts([query])[0]
    workflow_embs = emb_service.embed_texts(texts)
    scored = []
    for wf, emb in zip(workflows, workflow_embs):
        score = _cosine_similarity(query_emb, emb)
        scored.append({"workflow": wf, "score": round(score, 3)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def _search_workflows(db: Session, query: str, limit: int = 5):
    # Simple LIKE-based search over workflow name/description. Beginner-friendly.
    q = f"%{query}%"
    results = db.query(models.Workflow).filter(
        (models.Workflow.name.ilike(q)) | (models.Workflow.description.ilike(q))
    ).limit(limit).all()
    return results


def retrieve(query: str, top_k: int = 5):
    db = SessionLocal()
    try:
        chroma_res = emb_service.query(query, top_k=top_k)

        chunks = []
        try:
            docs = chroma_res.get("documents", [])
            metadatas = chroma_res.get("metadatas", [])
            distances = chroma_res.get("distances", [])
        except Exception:
            docs = []
            metadatas = []
            distances = []

        if docs:
            for idx, text in enumerate(docs[0] if docs and isinstance(docs[0], list) else docs):
                source = None
                chunk_id = None
                if metadatas and len(metadatas) > 0 and isinstance(metadatas[0], list) and idx < len(metadatas[0]):
                    metadata = metadatas[0][idx] or {}
                    source = metadata.get("filename")
                    chunk_id = metadata.get("document_id")
                score = 0.0
                if distances and len(distances) > 0 and isinstance(distances[0], list) and idx < len(distances[0]):
                    distance = distances[0][idx] or 0.0
                    score = max(0.0, round(1.0 - float(distance), 3))
                chunks.append(
                    {
                        "type": "policy",
                        "content": text,
                        "source": source,
                        "document_id": chunk_id,
                        "score": score,
                    }
                )

        workflows = _search_workflows(db, query, limit=top_k)
        workflows_scored = _workflow_scores(query, workflows)
        workflow_results = []
        for item in workflows_scored:
            wf = item["workflow"]
            workflow_results.append(
                {
                    "type": "workflow",
                    "title": wf.name,
                    "description": wf.description,
                    "steps": [step.description for step in wf.steps],
                    "score": item["score"],
                }
            )

        return {"chunks": chunks, "workflows": workflow_results}
    finally:
        db.close()
