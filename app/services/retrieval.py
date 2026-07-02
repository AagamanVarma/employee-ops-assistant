"""Retrieval over stored document chunks and workflows.

This module ranks document chunks and workflows with hybrid lexical and
semantic scoring, and returns debug data for inspection.
"""
import logging
import os
import re

import numpy as np
from dotenv import load_dotenv
from nltk.stem.porter import PorterStemmer

from app import models
from app.database import SessionLocal
from app.services.query_classifier import classify_query
from app.services.schema import ensure_schema
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
stemmer = PorterStemmer()
load_dotenv()

MIN_DOCUMENT_SCORE = 0.08
MIN_CHUNK_SCORE = 0.08
MIN_WORKFLOW_SCORE = 0.08


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s value %r, using default %s", name, value, default)
        return default


RETRIEVAL_THRESHOLD = min(max(_read_float_env("RETRIEVAL_THRESHOLD", 0.55), 0.0), 1.0)


class SemanticEncoder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
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


def _confidence_adjustment(
    query: str,
    text: str,
    base_score: float,
    *,
    title_text: str = "",
    text_weight: float = 0.45,
    title_weight: float = 0.18,
    weak_penalty: float = 0.12,
):
    query_tokens = _token_set(query)
    text_tokens = _token_set(text)
    title_tokens = _token_set(title_text) if title_text else set()
    matched_tokens = query_tokens & (text_tokens | title_tokens)
    matched_count = len(matched_tokens)
    query_token_count = len(query_tokens)

    text_overlap = _overlap_bonus(query, text)
    title_overlap = _overlap_bonus(query, title_text) if title_text else 0.0
    boosted = float(base_score) + (text_overlap * text_weight) + (title_overlap * title_weight)
    boosted += min(matched_count * 0.06, 0.12)

    if matched_count == 0:
        boosted -= weak_penalty
    elif query_token_count > 1 and matched_count < 2:
        boosted -= weak_penalty + 0.06
    elif max(text_overlap, title_overlap) < 0.15:
        boosted -= weak_penalty / 2

    boosted = max(0.0, min(boosted, 1.0))
    return {
        "score": round(boosted, 3),
        "raw_score": round(float(base_score), 3),
        "text_overlap": round(text_overlap, 3),
        "title_overlap": round(title_overlap, 3),
        "matched_count": matched_count,
        "query_token_count": query_token_count,
    }


def _lexical_scores(query: str, texts):
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

    lexical_scores = []
    query_terms = {term for term in re.findall(r"[a-z0-9]{2,}", _normalize(query))}
    for index, (word_score, char_score) in enumerate(zip(word_scores, char_scores)):
        lexical_score = 0.7 * float(word_score) + 0.3 * float(char_score)
        if query_terms:
            text_terms = {term for term in re.findall(r"[a-z0-9]{2,}", _normalize(texts[index]))}
            if query_terms & text_terms:
                lexical_score += 0.12
        lexical_scores.append(float(np.clip(lexical_score, 0.0, 1.0)))
    return lexical_scores


def _hybrid_score_details(query: str, texts):
    if not texts:
        return []
    semantic_scores = _semantic_scores(query, texts)
    lexical_scores = _lexical_scores(query, texts)
    details = []
    for semantic_score, lexical_score in zip(semantic_scores, lexical_scores):
        final_score = 0.55 * float(semantic_score) + 0.45 * float(lexical_score)
        details.append(
            {
                "semantic_score": round(float(semantic_score), 3),
                "lexical_score": round(float(lexical_score), 3),
                "hybrid_score": round(float(final_score), 3),
            }
        )
    return details


def _hybrid_scores(query: str, texts):
    score_details = _hybrid_score_details(query, texts)
    return [item["hybrid_score"] for item in score_details]


def _search_workflows(db: Session, query: str, limit: int = 5):
    candidates = db.query(models.Workflow).all()
    return candidates


def _workflow_scores(query: str, workflows, min_score: float = 0.0):
    if not workflows:
        return []
    texts = [f"{wf.name} {wf.description or ''} {' '.join(step.description for step in wf.workflow_steps)}" for wf in workflows]
    score_details = _hybrid_score_details(query, texts)
    scored = []
    for wf, details, text in zip(workflows, score_details, texts):
        adjusted = _confidence_adjustment(query, text, details["hybrid_score"], title_text=wf.name, text_weight=0.35, title_weight=0.25)
        boosted = adjusted["score"]
        if boosted <= max(0.02, min_score):
            continue
        scored.append(
            {
                "workflow": wf,
                "score": boosted,
                "raw_score": adjusted["raw_score"],
                "text_overlap": adjusted["text_overlap"],
                "title_overlap": adjusted["title_overlap"],
                "text": text,
                "semantic_score": details["semantic_score"],
                "lexical_score": details["lexical_score"],
                "hybrid_score": details["hybrid_score"],
                "citation": {
                    "workflow": wf.name,
                    "steps": len(wf.workflow_steps),
                },
            }
        )
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def _score_chunks(query: str, chunks, min_score: float = 0.0):
    if not chunks:
        return []
    texts = [f"{chunk.section_title or ''} {chunk.chunk_text}".strip() for chunk in chunks]
    score_details = _hybrid_score_details(query, texts)
    scored = []
    seen = set()
    for chunk, details in zip(chunks, score_details):
        display_text = chunk.chunk_text.strip()
        signature = _dedupe_key(display_text)
        if not signature or signature in seen:
            continue
        seen.add(signature)

        adjusted = _confidence_adjustment(
            query,
            f"{chunk.section_title or ''} {display_text}",
            details["hybrid_score"],
            title_text=chunk.section_title or chunk.filename,
            text_weight=0.45,
            title_weight=0.2,
        )
        boosted = adjusted["score"]
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
                "score": boosted,
                "semantic_score": details["semantic_score"],
                "lexical_score": details["lexical_score"],
                "hybrid_score": details["hybrid_score"],
                "raw_score": adjusted["raw_score"],
                "text_overlap": adjusted["text_overlap"],
                "title_overlap": adjusted["title_overlap"],
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
        adjusted = _confidence_adjustment(query, text, score, title_text=doc.filename, text_weight=0.28, title_weight=0.42)
        boosted = adjusted["score"]
        if boosted <= max(0.02, min_score):
            continue
        ranked.append(
            {
                "document": doc,
                "score": boosted,
                "raw_score": adjusted["raw_score"],
                "text_overlap": adjusted["text_overlap"],
                "title_overlap": adjusted["title_overlap"],
                "text": text,
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _top_candidate(items, label: str):
    if not items:
        return None
    top_item = max(items, key=lambda item: item["score"])
    return {
        "type": label,
        "score": top_item["score"],
        "raw_score": top_item.get("raw_score"),
        "text_overlap": top_item.get("text_overlap"),
        "title_overlap": top_item.get("title_overlap"),
    }


def _query_term_density(query: str, text: str) -> float:
    query_terms = [term for term in re.findall(r"[a-z0-9]{2,}", _normalize(query))]
    if not query_terms:
        return 0.0
    text_terms = [term for term in re.findall(r"[a-z0-9]{2,}", _normalize(text))]
    if not text_terms:
        return 0.0
    overlap = sum(1 for term in query_terms if term in text_terms)
    return overlap / max(len(query_terms), 1)


def _rerank_candidates(query: str, candidates: list[dict], *, title_field: str = "title") -> list[dict]:
    if not candidates:
        return []
    normalized_query = _normalize(query)
    query_terms = set(re.findall(r"[a-z0-9]{2,}", normalized_query))
    acronym_terms = {term for term in query_terms if len(term) <= 4}
    reranked = []
    for candidate in candidates:
        text = candidate.get("content") or candidate.get("text") or ""
        title = candidate.get("section_title") or candidate.get("title") or ""
        base_score = float(candidate.get("score", 0.0))
        exact_overlap = sum(1 for term in query_terms if term and term in _normalize(text))
        acronym_overlap = sum(1 for term in acronym_terms if term and term in _normalize(text))
        title_overlap = sum(1 for term in query_terms if term and term in _normalize(title))
        density = _query_term_density(query, text)
        boost = 0.0
        if exact_overlap:
            boost += 0.06 * min(exact_overlap, 3)
        if acronym_overlap:
            boost += 0.05 * min(acronym_overlap, 2)
        if title_overlap:
            boost += 0.05 * min(title_overlap, 2)
        if density > 0.2:
            boost += 0.04
        if candidate.get("section_title") and candidate.get("section_title") in title:
            boost += 0.01
        reranked_score = min(1.0, base_score + boost)
        reranked.append(
            {
                **candidate,
                "score": round(reranked_score, 3),
                "rerank_boost": round(boost, 3),
                "reranked_score": round(reranked_score, 3),
                "original_score": base_score,
            }
        )
    return sorted(reranked, key=lambda item: item["reranked_score"], reverse=True)


def _retrieval_limits(intent: str, top_k: int) -> dict[str, int]:
    if intent == "workflow":
        return {"doc_limit": 2, "chunk_limit": 2, "workflow_limit": max(top_k, 3)}
    if intent == "policy":
        return {"doc_limit": max(3, top_k), "chunk_limit": max(top_k, 3), "workflow_limit": 1}
    if intent == "mixed":
        return {"doc_limit": max(3, top_k), "chunk_limit": max(2, min(top_k, 3)), "workflow_limit": max(2, min(top_k, 3))}
    return {"doc_limit": 0, "chunk_limit": 0, "workflow_limit": 0}


def retrieve(query: str, top_k: int = 5):
    ensure_schema()
    db = SessionLocal()
    try:
        intent_result = classify_query(query)
        intent = intent_result.get("intent", "unknown")
        retrieval_limits = _retrieval_limits(intent, top_k)

        if intent == "unknown":
            fallback_reason = "Query intent was classified as unknown, so retrieval returned no policy or workflow context."
            debug = {
                "query": query,
                "threshold": RETRIEVAL_THRESHOLD,
                "top_score": 0.0,
                "top_candidate": None,
                "is_confident": False,
                "fallback_reason": fallback_reason,
                "intent_classification": intent_result,
                "retrieval_strategy": "unknown_fallback",
                "documents": [],
                "chunks": [],
                "workflows": [],
                "selected_documents": [],
                "selected_chunks": [],
                "selected_workflows": [],
            }
            logger.info("retrieval_debug intent=%s query=%r strategy=unknown_fallback", intent, query)
            return {
                "chunks": [],
                "workflows": [],
                "debug": debug,
                "confidence": {
                    "threshold": RETRIEVAL_THRESHOLD,
                    "top_score": 0.0,
                    "is_confident": False,
                    "has_retrieval_context": False,
                    "fallback_reason": fallback_reason,
                    "fallback_message": "I could not find a relevant match for this query in the uploaded company policies or workflows. Please contact HR for clarification.",
                },
            }

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
        ranked_docs = [item for item in ranked_docs_all if item["score"] >= MIN_DOCUMENT_SCORE][: max(retrieval_limits["doc_limit"], 1)]

        chunk_candidates = []
        for item in ranked_docs or ranked_docs_all[: max(retrieval_limits["doc_limit"], 1)]:
            doc = item["document"]
            doc_chunks = (
                db.query(models.DocumentChunk)
                .filter(models.DocumentChunk.document_id == doc.id)
                .order_by(models.DocumentChunk.chunk_index.asc())
                .all()
            )
            chunk_candidates.extend(doc_chunks)

        chunk_ranked_all = _score_chunks(query, chunk_candidates, min_score=0.0)
        broadened_chunks = [item for item in chunk_ranked_all if item["score"] >= MIN_CHUNK_SCORE]
        reranked_chunks = _rerank_candidates(query, broadened_chunks)
        chunks = reranked_chunks[: retrieval_limits["chunk_limit"]]

        workflows = _search_workflows(db, query, limit=top_k)
        workflow_ranked_all = _workflow_scores(query, workflows, min_score=0.0)
        broadened_workflows = [item for item in workflow_ranked_all if item["score"] >= MIN_WORKFLOW_SCORE]
        reranked_workflows = _rerank_candidates(query, broadened_workflows, title_field="title")
        workflow_results = []
        for item in reranked_workflows[: retrieval_limits["workflow_limit"]]:
            wf = item["workflow"]
            workflow_results.append(
                {
                    "workflow": wf,
                    "type": "workflow",
                    "title": wf.name,
                    "description": wf.description,
                    "steps": [step.description for step in wf.workflow_steps],
                    "score": item["score"],
                    "rerank_boost": item.get("rerank_boost"),
                    "reranked_score": item.get("reranked_score"),
                    "semantic_score": item.get("semantic_score"),
                    "lexical_score": item.get("lexical_score"),
                    "hybrid_score": item.get("hybrid_score"),
                }
            )

        top_candidates = [candidate for candidate in [
            _top_candidate(chunks, "chunk"),
            _top_candidate(workflow_results, "workflow"),
        ] if candidate is not None]
        top_candidate = max(top_candidates, key=lambda item: item["score"], default=None)
        top_score = top_candidate["score"] if top_candidate else 0.0
        has_retrieval_context = bool(chunks or workflow_results)
        is_confident = top_score >= RETRIEVAL_THRESHOLD and bool(top_candidates)

        if not is_confident and not has_retrieval_context:
            chunks = []
            workflow_results = []

        fallback_reason = None
        if not is_confident:
            if top_candidate is None:
                fallback_reason = "No chunk or workflow cleared the retrieval filters."
            else:
                fallback_reason = (
                    f"Top {top_candidate['type']} score {top_score:.3f} is below the threshold {RETRIEVAL_THRESHOLD:.3f}."
                )

        debug = {
            "query": query,
            "threshold": RETRIEVAL_THRESHOLD,
            "top_score": round(top_score, 3),
            "top_candidate": top_candidate,
            "is_confident": is_confident,
            "fallback_reason": fallback_reason,
            "intent_classification": intent_result,
            "retrieval_strategy": f"{intent}_intent",
            "workflow_count": len(workflow_results),
            "chunk_count": len(chunks),
            "documents": [
                {
                    "name": item["document"].filename,
                    "score": item["score"],
                    "raw_score": item.get("raw_score"),
                    "text_overlap": item.get("text_overlap"),
                    "title_overlap": item.get("title_overlap"),
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
                    "raw_score": item.get("raw_score"),
                    "text_overlap": item.get("text_overlap"),
                    "title_overlap": item.get("title_overlap"),
                    "semantic_score": item.get("semantic_score"),
                    "lexical_score": item.get("lexical_score"),
                    "hybrid_score": item.get("hybrid_score"),
                    "rerank_boost": item.get("rerank_boost"),
                    "reranked_score": item.get("reranked_score"),
                    "snippet": item["content"][:280],
                }
                for item in chunks
            ],
            "workflows": [
                {
                    "title": item["workflow"].name,
                    "score": item["score"],
                    "raw_score": item.get("raw_score"),
                    "text_overlap": item.get("text_overlap"),
                    "title_overlap": item.get("title_overlap"),
                    "semantic_score": item.get("semantic_score"),
                    "lexical_score": item.get("lexical_score"),
                    "hybrid_score": item.get("hybrid_score"),
                    "rerank_boost": item.get("rerank_boost"),
                    "reranked_score": item.get("reranked_score"),
                    "snippet": (item.get("text") or "")[:280],
                }
                for item in workflow_results
            ],
            "selected_documents": [
                {
                    "name": item["document"].filename,
                    "score": item["score"],
                    "raw_score": item.get("raw_score"),
                }
                for item in ranked_docs[:top_k]
            ],
            "selected_chunks": [
                {
                    "source": item["source"],
                    "section_title": item.get("section_title"),
                    "page_number": item.get("page_number"),
                    "score": item["score"],
                    "raw_score": item.get("raw_score"),
                    "rerank_boost": item.get("rerank_boost"),
                    "reranked_score": item.get("reranked_score"),
                }
                for item in chunks
            ],
            "selected_workflows": [
                {
                    "title": item["workflow"].name,
                    "score": item["score"],
                    "raw_score": item.get("raw_score"),
                    "rerank_boost": item.get("rerank_boost"),
                    "reranked_score": item.get("reranked_score"),
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

        return {
            "chunks": chunks,
            "workflows": workflow_results,
            "debug": debug,
            "confidence": {
                "threshold": RETRIEVAL_THRESHOLD,
                "top_score": round(top_score, 3),
                "is_confident": is_confident,
                "has_retrieval_context": has_retrieval_context,
                "fallback_reason": fallback_reason,
                "fallback_message": "Please contact HR for clarification." if not is_confident and not has_retrieval_context else None,
            },
        }
    finally:
        db.close()
