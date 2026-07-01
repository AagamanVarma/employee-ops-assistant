"""Embeddings and ChromaDB helpers using sentence-transformers and chromadb.

This module is intentionally lazy: it avoids loading the SentenceTransformer model
and initializing Chroma until semantic embedding or search is actually needed.
"""
import logging
import chromadb
from chromadb.config import Settings
from pathlib import Path

CHROMA_DIR = Path("data") / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingsService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.client = None
        self.collection = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            logging.error("Failed to load SentenceTransformer: %s", exc, exc_info=True)
            raise

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            self.client = chromadb.Client(Settings(persist_directory=str(CHROMA_DIR), is_persistent=True))
            return self.client
        except Exception as exc:
            logging.error("Failed to initialize Chroma client: %s", exc, exc_info=True)
            raise

    def get_or_create_collection(self, name: str = "documents"):
        if self.collection is None:
            client = self._get_client()
            self.collection = client.get_or_create_collection(name=name)
        return self.collection

    def embed_texts(self, texts):
        self._load_model()
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def add_documents(self, ids, texts, metadatas=None, collection_name: str = "documents"):
        coll = self.get_or_create_collection(collection_name)
        embeddings = self.embed_texts(texts)
        coll.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts], embeddings=embeddings)
        self.client.persist()

    def query(self, query_text, top_k: int = 5, collection_name: str = "documents"):
        coll = self.get_or_create_collection(collection_name)
        q_emb = self.embed_texts([query_text])[0]
        res = coll.query(query_embeddings=[q_emb], n_results=top_k)
        return res


# Module-level shared embeddings service for both indexing and retrieval.
emb_service = EmbeddingsService()
