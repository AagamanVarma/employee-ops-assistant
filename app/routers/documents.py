from fastapi import APIRouter, Request, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import logging

from app.database import get_db, engine
from app import models
from app.routers.admin import get_current_admin
from app.services.document_processor import process_document_background
from app.services.embeddings import emb_service

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/admin/documents")
def list_documents(request: Request, db: Session = Depends(get_db)):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")
    docs = db.query(models.Document).order_by(models.Document.uploaded_at.desc()).all()
    return templates.TemplateResponse(request, "admin/documents.html", {"request": request, "documents": docs})


@router.get("/admin/documents/upload")
def upload_form(request: Request):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")
    return templates.TemplateResponse(request, "admin/upload_document.html", {"request": request})


@router.post("/admin/documents/upload")
def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")

    filename = Path(file.filename).name
    dest = DOCS_DIR / filename
    # Avoid overwriting by appending a counter if necessary
    counter = 1
    base = dest.stem
    suffix = dest.suffix
    while dest.exists():
        dest = DOCS_DIR / f"{base}-{counter}{suffix}"
        counter += 1

    with dest.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    doc = models.Document(filename=filename, filepath=str(dest))
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(process_document_background, doc.id)

    return RedirectResponse(url="/admin/documents", status_code=303)


@router.post("/admin/documents/delete/{doc_id}")
def delete_document(
    request: Request,
    doc_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return RedirectResponse(url="/admin/documents")

    # collect vector IDs for precise Chroma cleanup before deleting DB rows
    vector_ids = [
        row.vector_id
        for (row,) in db.query(models.DocumentChunk.vector_id)
        .filter(models.DocumentChunk.document_id == doc.id)
        .all()
        if row
    ]

    # delete related chunks and document row quickly
    try:
        db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == doc.id).delete()
        db.delete(doc)
        db.commit()
        logger.debug("Deleted document %s and its chunks", doc.id)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete document %s or its chunks: %s", doc.id, exc, exc_info=True)
        return RedirectResponse(url="/admin/documents", status_code=303)

    # delete physical file safely after database commit
    try:
        file_path = Path(doc.filepath)
        if file_path.exists():
            file_path.unlink()
            logger.debug("Deleted document file %s", file_path)
        else:
            logger.debug("Document file not found during deletion: %s", file_path)
    except Exception as exc:
        logger.warning("Failed to delete document file %s: %s", doc.filepath, exc, exc_info=True)

    # schedule Chroma cleanup in FastAPI background
    def _cleanup_doc_vectors(ids: list[str]):
        try:
            result = emb_service.delete_vector_ids(ids)
            logger.debug("Background Chroma cleanup finished for document %s ids=%s: %s", doc.id, ids, result)
        except Exception as exc:
            logger.warning(
                "Background Chroma cleanup failed for document %s ids=%s: %s",
                doc.id,
                ids,
                exc,
                exc_info=True,
            )

    background_tasks.add_task(_cleanup_doc_vectors, vector_ids)
    logger.debug("Scheduled Chroma cleanup background task for document %s", doc.id)

    return RedirectResponse(url="/admin/documents", status_code=303)
