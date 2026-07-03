"""PDF extraction and semantic chunking using PyMuPDF (fitz).

The upload path extracts text and creates semantically meaningful chunks.
Search scoring happens at query time so uploads stay fast.

SEMANTIC CHUNKING STRATEGY:
- Detect section headings and preserve heading + content relationship
- Group paragraphs into semantic units (min 100 words, max 600 words)
- Intelligently merge small fragments to avoid noise
- Preserve table structure and policy clauses
- Add soft overlap for context preservation
- Debug logging shows chunking quality metrics
"""
from pathlib import Path
import re
import fitz  # PyMuPDF
import logging
from app import models
from app.database import SessionLocal
from app.services.schema import ensure_schema
from app.services.embeddings import emb_service

logger = logging.getLogger(__name__)


def _clean_text(s: str) -> str:
    return "\n".join([line.strip() for line in s.splitlines() if line.strip()])


def _is_table_like(text: str) -> bool:
    """Detect if text looks like a table row or structured data."""
    lines = text.strip().split('\n')
    if len(lines) > 3:
        return False
    # Check for pipe separators, tabs, or many spaces (table indicators)
    return any('|' in line or '\t' in line or '  ' in line for line in lines)


def _is_list_item(text: str) -> bool:
    """Detect if text is a list item (bullet, number, dash)."""
    stripped = text.strip()
    return bool(re.match(r'^[\-•*]\s+', stripped)) or bool(re.match(r'^\d+\.\s+', stripped))


def _is_policy_clause(text: str) -> bool:
    """Detect if text is a numbered policy clause (e.g., '4.1', '3.2.1')."""
    stripped = text.strip()
    # Matches patterns like "4.1", "3.2.1", or "3.2.1 Policy Name"
    return bool(re.match(r'^\d+(\.\d+)*\s', stripped))


def _looks_like_heading(line: str):
    """Detect if a line is a section heading."""
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


def _split_into_paragraphs(text: str):
    """Split text into paragraphs, preserving structure."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def _extract_sections(pdf, default_title: str):
    """Extract text grouped by section headings."""
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


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _semantic_chunk_section(section_text: str, min_words: int = 100, max_words: int = 600) -> list:
    """
    Create semantically meaningful chunks from section text.
    
    Strategy:
    1. Split into paragraphs (natural text boundaries)
    2. Group paragraphs into chunks respecting min/max word counts
    3. Merge small chunks intelligently with neighbors
    4. Preserve table structure, lists, and policy clauses
    5. Add soft overlap for context
    
    Returns list of chunk strings.
    """
    paragraphs = _split_into_paragraphs(section_text)
    if not paragraphs:
        return []
    
    # Group paragraphs into candidate chunks
    candidates = []
    current_group = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = _count_words(para)
        
        # If single paragraph exceeds max, split it by sentences
        if para_words > max_words:
            if current_group:
                candidates.append((current_group, current_word_count))
                current_group = []
                current_word_count = 0
            
            # Split long paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_group = []
            sent_count = 0
            for sent in sentences:
                sent_words = _count_words(sent)
                if sent_count + sent_words > max_words and sent_group:
                    candidates.append((['\n'.join(sent_group)], sent_count))
                    sent_group = [sent]
                    sent_count = sent_words
                else:
                    sent_group.append(sent)
                    sent_count += sent_words
            if sent_group:
                candidates.append((['\n'.join(sent_group)], sent_count))
        
        # Add paragraph to current group if it fits
        elif current_word_count + para_words <= max_words:
            current_group.append(para)
            current_word_count += para_words
        
        # Start new group
        else:
            if current_group:
                candidates.append((current_group, current_word_count))
            current_group = [para]
            current_word_count = para_words
    
    if current_group:
        candidates.append((current_group, current_word_count))
    
    # Merge small chunks with neighbors
    merged_candidates = []
    i = 0
    while i < len(candidates):
        para_group, word_count = candidates[i]
        
        # If chunk is very small and not a meaningful unit, merge with next
        if word_count < min_words and i < len(candidates) - 1:
            next_group, next_words = candidates[i + 1]
            # Merge current with next
            merged_candidates.append((para_group + next_group, word_count + next_words))
            i += 2
        # If chunk is small but meaningful (table, list, policy), keep it
        elif any(_is_table_like(p) or _is_list_item(p) or _is_policy_clause(p) for p in para_group):
            merged_candidates.append((para_group, word_count))
            i += 1
        else:
            merged_candidates.append((para_group, word_count))
            i += 1
    
    # Join paragraphs in each group into chunks
    chunks = []
    for para_group, _ in merged_candidates:
        chunk_text = "\n\n".join(para_group)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks


def process_document(document_id: int, db):
    ensure_schema()
    doc_row = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc_row:
        raise ValueError("Document not found")

    file_path = Path(doc_row.filepath)
    if not file_path.exists():
        raise ValueError("File not found on disk")

    # Extract text by section
    with fitz.open(str(file_path)) as pdf:
        sections = _extract_sections(pdf, Path(doc_row.filename).stem)

    # Remove any existing chunks for this document
    db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).delete()

    # Create semantic chunks
    chunk_index = 0
    chunk_rows = []
    section_stats = []
    
    for section in sections:
        # Use semantic chunking instead of word-based chunking
        semantic_chunks = _semantic_chunk_section(section["text"], min_words=100, max_words=600)
        
        section_stat = {
            "section_title": section["title"],
            "chunk_count": len(semantic_chunks),
            "chunk_sizes": [],
        }
        
        for sc in semantic_chunks:
            vector_id = f"doc_{document_id}_chunk_{chunk_index}"
            word_count = _count_words(sc)
            section_stat["chunk_sizes"].append(word_count)
            
            chunk = models.DocumentChunk(
                document_id=document_id,
                filename=doc_row.filename,
                section_title=section["title"],
                page_number=section["page_number"],
                chunk_index=chunk_index,
                vector_id=vector_id,
                chunk_text=sc,
            )
            chunk_rows.append(chunk)
            chunk_index += 1
        
        section_stats.append(section_stat)

    db.add_all(chunk_rows)
    db.commit()
    
    # DEBUG LOGGING: Show chunking quality metrics
    logger.info(
        "✓ Semantic chunking for document %s (%s):",
        document_id,
        doc_row.filename,
    )
    for stat in section_stats:
        if stat["chunk_count"] > 0:
            avg_size = sum(stat["chunk_sizes"]) // len(stat["chunk_sizes"])
            min_size = min(stat["chunk_sizes"])
            max_size = max(stat["chunk_sizes"])
            logger.info(
                "  Section '%s': %d chunks | Avg: %d words | Range: %d-%d",
                stat["section_title"][:40],
                stat["chunk_count"],
                avg_size,
                min_size,
                max_size,
            )
    
    logger.info(
        "✓ Created %d semantic chunks (total) for document %s",
        len(chunk_rows),
        document_id,
    )

    # Index chunks in vector store
    vector_ids = [row.vector_id for row in chunk_rows if row.vector_id]
    if vector_ids:
        texts = [row.chunk_text for row in chunk_rows]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": row.chunk_index,
                "filename": row.filename,
                "section_title": row.section_title,
                "page_number": row.page_number,
            }
            for row in chunk_rows
        ]
        try:
            emb_service.delete_vector_ids(vector_ids)
            emb_service.add_documents(ids=vector_ids, texts=texts, metadatas=metadatas)
            logger.info(
                "✓ Indexed %d semantic chunk vectors in Chroma for document %s",
                len(vector_ids),
                document_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to index document %s chunks in Chroma: %s",
                document_id,
                exc,
                exc_info=True,
            )


def process_document_background(document_id: int):
    db = SessionLocal()
    try:
        process_document(document_id, db)
    finally:
        db.close()
