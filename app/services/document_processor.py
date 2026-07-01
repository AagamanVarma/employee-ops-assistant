"""PDF extraction and chunking using PyMuPDF (fitz).

The upload path only extracts text and stores chunks. Search scoring happens at
query time so uploads stay fast.
"""
from pathlib import Path
import re
import fitz  # PyMuPDF
from app import models
from app.database import SessionLocal
from app.services.schema import ensure_schema


def _clean_text(s: str) -> str:
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])


def _split_into_paragraphs(text: str):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def _split_into_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _looks_like_heading(line: str):
    clean = re.sub(r"^[\-•*\d.\s]+", "", line).strip()
    if not clean or len(clean) > 90:
        return False
    if clean.endswith((".", "!", "?")):
        return False
    words = clean.split()
    if len(words) > 10:
        return False
    if clean.endswith(":"):
        return True
    if clean.isupper():
        return True
    title_words = sum(1 for word in words if word[:1].isupper())
    return title_words >= max(2, len(words) - 1)


def _extract_sections(pdf, default_title: str):
    sections = []
    current_title = default_title
    current_page = 1
    current_lines = []

    def flush_section():
        nonlocal current_lines, current_title, current_page
        text = _clean_text("\n".join(current_lines))
        if text:
            sections.append(
                {
                    "title": current_title,
                    "page_number": current_page,
                    "text": text,
                }
            )
        current_lines = []

    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text("text") or ""
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue
            if _looks_like_heading(line):
                if current_lines:
                    flush_section()
                current_title = line.rstrip(":").strip()
                current_page = page_number
                continue
            current_lines.append(line)

    if current_lines:
        flush_section()

    return sections


def _chunk_words(text: str, max_words: int = 420, overlap_words: int = 80):
    paragraphs = _split_into_paragraphs(text)
    chunks = []
    buffer_words = []

    def flush_buffer():
        nonlocal buffer_words
        if buffer_words:
            chunks.append(" ".join(buffer_words).strip())
            if overlap_words > 0:
                buffer_words = buffer_words[-overlap_words:]
            else:
                buffer_words = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        if len(words) > max_words:
            flush_buffer()
            sentence_buffer = []
            sentence_words = 0
            for sentence in _split_into_sentences(paragraph):
                sentence_parts = sentence.split()
                if sentence_buffer and sentence_words + len(sentence_parts) > max_words:
                    chunks.append(" ".join(sentence_buffer).strip())
                    sentence_buffer = sentence_buffer[-overlap_words:] if overlap_words > 0 else []
                    sentence_words = len(sentence_buffer)
                sentence_buffer.extend(sentence_parts)
                sentence_words += len(sentence_parts)
            if sentence_buffer:
                chunks.append(" ".join(sentence_buffer).strip())
            buffer_words = []
            continue

        if buffer_words and len(buffer_words) + len(words) > max_words:
            flush_buffer()
        buffer_words.extend(words)

    if buffer_words:
        chunks.append(" ".join(buffer_words).strip())

    return [chunk for chunk in chunks if chunk]


def process_document(document_id: int, db):
    ensure_schema()
    doc_row = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc_row:
        raise ValueError("Document not found")

    file_path = Path(doc_row.filepath)
    if not file_path.exists():
        raise ValueError("File not found on disk")

    # extract text
    with fitz.open(str(file_path)) as pdf:
        sections = _extract_sections(pdf, Path(doc_row.filename).stem)

    # remove any existing chunks for this document
    db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).delete()

    chunk_index = 0
    for section in sections:
        subchunks = _chunk_words(section["text"])
        for sc in subchunks:
            chunk = models.DocumentChunk(
                document_id=document_id,
                filename=doc_row.filename,
                section_title=section["title"],
                page_number=section["page_number"],
                chunk_index=chunk_index,
                chunk_text=sc,
            )
            db.add(chunk)
            chunk_index += 1

    db.commit()

    # Search uses the stored chunks directly, so there is no separate indexing step here.


def process_document_background(document_id: int):
    db = SessionLocal()
    try:
        process_document(document_id, db)
    finally:
        db.close()
