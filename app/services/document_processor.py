"""Simple PDF extraction and chunking using PyMuPDF (fitz).

Functions:
- process_document(document_id, db): extracts text, cleans, and stores chunks in DB.

Notes:
- Keeps chunking simple: split by double-newline paragraphs, and further split long paragraphs into ~1000-char chunks.
"""
from pathlib import Path
import fitz  # PyMuPDF
from app import models
from app.services.embeddings import emb_service


def _clean_text(s: str) -> str:
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])


def _split_into_paragraphs(text: str):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def _chunk_paragraph(paragraph: str, max_chars: int = 1000):
    if len(paragraph) <= max_chars:
        return [paragraph]
    chunks = []
    start = 0
    while start < len(paragraph):
        chunks.append(paragraph[start : start + max_chars])
        start += max_chars
    return chunks


def process_document(document_id: int, db):
    doc_row = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc_row:
        raise ValueError("Document not found")

    file_path = Path(doc_row.filepath)
    if not file_path.exists():
        raise ValueError("File not found on disk")

    # extract text
    text_parts = []
    with fitz.open(str(file_path)) as pdf:
        for page in pdf:
            text = page.get_text()
            if text:
                text_parts.append(text)

    full_text = _clean_text("\n\n".join(text_parts))
    paragraphs = _split_into_paragraphs(full_text)

    # remove any existing chunks for this document
    db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).delete()

    chunk_index = 0
    for p in paragraphs:
        subchunks = _chunk_paragraph(p)
        for sc in subchunks:
            chunk = models.DocumentChunk(
                document_id=document_id,
                filename=doc_row.filename,
                chunk_index=chunk_index,
                chunk_text=sc,
            )
            db.add(chunk)
            chunk_index += 1

    db.commit()

    # index policy chunks into ChromaDB for semantic search
    chunk_rows = db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).order_by(models.DocumentChunk.chunk_index).all()
    if chunk_rows:
        ids = [f"{document_id}:{chunk.chunk_index}" for chunk in chunk_rows]
        texts = [chunk.chunk_text for chunk in chunk_rows]
        metadatas = [
            {"document_id": chunk.document_id, "filename": chunk.filename, "chunk_index": chunk.chunk_index}
            for chunk in chunk_rows
        ]
        emb_service.add_documents(ids=ids, texts=texts, metadatas=metadatas)
