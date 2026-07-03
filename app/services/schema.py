"""Lightweight SQLite schema helpers for additive columns."""
from app.database import engine


def _existing_columns(table_name: str):
    with engine.begin() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def ensure_schema():
    existing = _existing_columns("document_chunks")
    with engine.begin() as conn:
        if "section_title" not in existing:
            conn.exec_driver_sql("ALTER TABLE document_chunks ADD COLUMN section_title TEXT")
        if "page_number" not in existing:
            conn.exec_driver_sql("ALTER TABLE document_chunks ADD COLUMN page_number INTEGER")
        if "vector_id" not in existing:
            conn.exec_driver_sql("ALTER TABLE document_chunks ADD COLUMN vector_id TEXT")

    existing_workflow = _existing_columns("workflows")
    with engine.begin() as conn:
        if "vector_id" not in existing_workflow:
            conn.exec_driver_sql("ALTER TABLE workflows ADD COLUMN vector_id TEXT")