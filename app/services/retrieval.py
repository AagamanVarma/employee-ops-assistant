"""Retrieval over stored document chunks and workflows.

This module ranks document chunks and workflows with hybrid lexical and
semantic scoring, and returns debug data for inspection.
"""
import logging
import os
import re

import numpy as np
from nltk.stem.porter import PorterStemmer

from app import models
from app.database import SessionLocal
from app.services.schema import ensure_schema
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
stemmer = PorterStemmer()

MIN_DOCUMENT_SCORE = 0.08
MIN_CHUNK_SCORE = 0.08
MIN_WORKFLOW_SCORE = 0.08


class SemanticEncoder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self._model = None
        self._available = None

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic encoder unavailable, falling back to lexical scoring: %s", exc)
            self._available = False
        return self._available

    def embed(self, texts):
        if not self._load():
            return None
        vectors = self._model.encode(texts, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


semantic_encoder = SemanticEncoder()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _token_set(text: str):
    tokens = re.findall(r"[a-z0-9]{2,}", _normalize(text))
    return {stemmer.stem(token) for token in tokens}


def _tokenize(text: str):
    tokens = re.findall(r"[a-z0-9]{2,}", _normalize(text))
    return [stemmer.stem(token) for token in tokens]


def _overlap_bonus(query: str, text: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    text_tokens = _token_set(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / max(len(query_tokens), 1)


def _dedupe_key(text: str):
    normalized = _normalize(text)
    return re.sub(r"[^a-z0-9]+", " ", normalized[:240]).strip()


def _semantic_scores(query: str, texts):
    query_vec = semantic_encoder.embed([query])
    text_vecs = semantic_encoder.embed(texts)
    if query_vec is None or text_vecs is None:
        return np.zeros(len(texts), dtype=np.float32)
    return np.dot(text_vecs, query_vec[0])


def _hybrid_scores(query: str, texts):
    if not texts:
        return []
    word_vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        preprocessor=None,
        token_pattern=None,
        ngram_range=(1, 2),
    )
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    word_matrix = word_vectorizer.fit_transform([query] + texts)
    char_matrix = char_vectorizer.fit_transform([query] + texts)

    word_scores = (word_matrix[1:] * word_matrix[0].T).toarray().ravel()
    char_scores = (char_matrix[1:] * char_matrix[0].T).toarray().ravel()
    semantic_scores = _semantic_scores(query, texts)

    combined_scores = []
    for word_score, char_score, semantic_score in zip(word_scores, char_scores, semantic_scores):
        combined_scores.append(0.4 * word_score + 0.25 * char_score + 0.35 * float(semantic_score))
    return combined_scores


def _search_workflows(db: Session, query: str, limit: int = 5):
    candidates = db.query(models.Workflow).all()
    return candidates


def _workflow_scores(query: str, workflows, min_score: float = 0.0):
    if not workflows:
        return []
    texts = [f"{wf.name} {wf.description or ''} {' '.join(step.description for step in wf.steps)}" for wf in workflows]
    scores = _hybrid_scores(query, texts)
    scored = []
    for wf, score, text in zip(workflows, scores, texts):
        boosted = float(score) + _overlap_bonus(query, text) * 0.4
        if boosted <= max(0.02, min_score):
            continue
        scored.append(
            {
                "workflow": wf,
                "score": round(min(boosted, 1.0), 3),
                "text": text,
                "citation": {
                    "workflow": wf.name,
                    "steps": len(wf.steps),
                },
            }
        )
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def _score_chunks(query: str, chunks, min_score: float = 0.0):
    if not chunks:
        return []
    texts = [f"{chunk.section_title or ''} {chunk.chunk_text}".strip() for chunk in chunks]
    scores = _hybrid_scores(query, texts)
    scored = []
    seen = set()
    for chunk, score in zip(chunks, scores):
        display_text = chunk.chunk_text.strip()
        signature = _dedupe_key(display_text)
        if not signature or signature in seen:
            continue
        seen.add(signature)

        boosted = float(score) + _overlap_bonus(query, f"{chunk.section_title or ''} {display_text}") * 0.45
        if boosted <= max(0.02, min_score):
            continue
        scored.append(
            {
                "type": "policy",
                "content": display_text,
                "source": chunk.filename,
                "section_title": chunk.section_title,
                "page_number": chunk.page_number,
                "document_id": chunk.document_id,
                "score": round(min(boosted, 1.0), 3),
                "citation": {
                    "document": chunk.filename,
                    "section": chunk.section_title,
                    "page": chunk.page_number,
                },
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _score_documents(query: str, documents, min_score: float = 0.0):
    if not documents:
        return []
    texts = [f"{doc.filename} {doc_text}" for doc, doc_text in documents]
    scores = _hybrid_scores(query, texts)
    ranked = []
    for (doc, _), score, text in zip(documents, scores, texts):
        filename_bonus = _overlap_bonus(query, doc.filename) * 0.6
        doc_bonus = _overlap_bonus(query, text) * 0.25
        boosted = float(score) + filename_bonus + doc_bonus
        if boosted <= max(0.02, min_score):
            continue
        ranked.append({"document": doc, "score": round(min(boosted, 1.0), 3), "text": text})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def retrieve(query: str, top_k: int = 5):
    ensure_schema()
    db = SessionLocal()
    try:
        documents = db.query(models.Document).order_by(models.Document.uploaded_at.desc()).all()
        doc_text_map = []
        for doc in documents:
            doc_chunks = (
                db.query(models.DocumentChunk)
                .filter(models.DocumentChunk.document_id == doc.id)
                .order_by(models.DocumentChunk.chunk_index.asc())
                .all()
            )
            combined_text = " ".join(
                f"{chunk.section_title or ''} {chunk.chunk_text}".strip() for chunk in doc_chunks[:10]
            )
            doc_text_map.append((doc, combined_text))

        ranked_docs_all = _score_documents(query, doc_text_map, min_score=0.0)
        ranked_docs = [item for item in ranked_docs_all if item["score"] >= MIN_DOCUMENT_SCORE][: max(top_k, 5)]

        chunk_candidates = []
        for item in ranked_docs or ranked_docs_all[: max(top_k, 5)]:
            doc = item["document"]
            doc_chunks = (
                db.query(models.DocumentChunk)
                .filter(models.DocumentChunk.document_id == doc.id)
                .order_by(models.DocumentChunk.chunk_index.asc())
                .all()
            )
            chunk_candidates.extend(doc_chunks)

        chunk_ranked_all = _score_chunks(query, chunk_candidates, min_score=0.0)
        chunks = [item for item in chunk_ranked_all if item["score"] >= MIN_CHUNK_SCORE][:top_k]

        workflows = _search_workflows(db, query, limit=top_k)
        workflow_ranked_all = _workflow_scores(query, workflows, min_score=0.0)
        workflow_results = []
        for item in [item for item in workflow_ranked_all if item["score"] >= MIN_WORKFLOW_SCORE][:top_k]:
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

        debug = {
            "query": query,
            "documents": [
                {
                    "name": item["document"].filename,
                    "score": item["score"],
                    "document_id": item["document"].id,
                }
                for item in ranked_docs_all[:top_k]
            ],
            "chunks": [
                {
                    "source": item["source"],
                    "section_title": item.get("section_title"),
                    "page_number": item.get("page_number"),
                    "score": item["score"],
                    "snippet": item["content"][:280],
                }
                for item in chunk_ranked_all[:top_k]
            ],
            "workflows": [
                {
                    "title": item["workflow"].name,
                    "score": item["score"],
                    "snippet": item["text"][:280],
                }
                for item in workflow_ranked_all[:top_k]
            ],
            "selected_documents": [
                {
                    "name": item["document"].filename,
                    "score": item["score"],
                }
                for item in ranked_docs[:top_k]
            ],
            "selected_chunks": [
                {
                    "source": item["source"],
                    "section_title": item.get("section_title"),
                    "page_number": item.get("page_number"),
                    "score": item["score"],
                }
                for item in chunks
            ],
            "selected_workflows": [
                {
                    "title": item["workflow"].name,
                    "score": item["score"],
                }
                for item in workflow_results
            ],
        }

        logger.info(
            "retrieval_debug query=%r documents=%s chunks=%s workflows=%s",
            query,
            debug["documents"],
            debug["chunks"],
            debug["workflows"],
        )

        return {"chunks": chunks, "workflows": workflow_results, "debug": debug}
    finally:
        db.close()
