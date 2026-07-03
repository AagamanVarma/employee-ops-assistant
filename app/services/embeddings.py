"""Local embeddings and ChromaDB helpers.

This version keeps the app working without a large model download.
It uses a simple stateless hashing vectorizer for document and query embeddings.
"""
from pathlib import Path
import logging

import chromadb
from chromadb.config import Settings
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)
CHROMA_DIR = Path("data") / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingsService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            stop_words="english",
        )

    def _get_client(self):
        if self.client is None:
            self.client = chromadb.Client(
                Settings(
                    persist_directory=str(CHROMA_DIR),
                    is_persistent=True,
                )
            )
        return self.client

    def get_or_create_collection(self, name: str = "documents"):
        if self.collection is None:
            client = self._get_client()
            self.collection = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        return self.collection

    def embed_texts(self, texts):
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def add_documents(self, ids, texts, metadatas=None, collection_name: str = "documents"):
        coll = self.get_or_create_collection(collection_name)
        embeddings = self.embed_texts(texts)
        coll.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts], embeddings=embeddings)

    def query(self, query_text, top_k: int = 5, collection_name: str = "documents"):
        coll = self.get_or_create_collection(collection_name)
        q_emb = self.embed_texts([query_text])[0]
        return coll.query(query_embeddings=[q_emb], n_results=top_k)

    def delete_records(
        self,
        ids: list | None = None,
        where: dict | None = None,
        where_document: dict | None = None,
        collection_name: str = "documents",
    ) -> dict[str, int]:
        coll = self.get_or_create_collection(collection_name)
        if not ids and not where and not where_document:
            logger.debug("No Chroma deletion filter provided for collection %s", collection_name)
            return {"deleted_count": 0}

        if ids:
            ids = [str(i) for i in ids]

        try:
            result = coll.delete(ids=ids, where=where, where_document=where_document)
            deleted_count = 0
            if isinstance(result, dict):
                deleted_count = int(result.get("deleted_count", 0))
            else:
                deleted_count = int(getattr(result, "deleted_count", 0))
            logger.debug(
                "Deleted %d records from Chroma collection %s",
                deleted_count,
                collection_name,
            )
            return {"deleted_count": deleted_count}
        except Exception as exc:
            logger.warning(
                "Chroma deletion failed for collection %s: %s",
                collection_name,
                exc,
                exc_info=True,
            )
            return {"deleted_count": 0}

    def delete_document_vectors(self, document_id: int, collection_name: str = "documents") -> dict[str, int]:
        return self.delete_records(where={"document_id": document_id}, collection_name=collection_name)

    def delete_workflow_vectors(self, workflow_id: int, collection_name: str = "documents") -> dict[str, int]:
        return self.delete_records(where={"workflow_id": workflow_id}, collection_name=collection_name)


# Module-level shared embeddings service for both indexing and retrieval.
emb_service = EmbeddingsService()
