"""Local embeddings and ChromaDB helpers.

This version keeps the app working without a large model download.
It uses a simple stateless hashing vectorizer for document and query embeddings.
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sklearn.feature_extraction.text import HashingVectorizer

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


# Module-level shared embeddings service for both indexing and retrieval.
emb_service = EmbeddingsService()
