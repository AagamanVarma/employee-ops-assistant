from fastapi import APIRouter, Request, Depends, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from pathlib import Path
import shutil

from app.database import get_db, engine
from app import models
from app.routers.admin import get_current_admin
from app.services.document_processor import process_document

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
def upload_document(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
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

    # Process document (extract + chunk) asynchronously would be ideal; keep simple and synchronous for PoC
    try:
        process_document(doc.id, db)
    except Exception as e:
        # log error but continue
        print("Error processing document:", e)

    return RedirectResponse(url="/admin/documents", status_code=303)


@router.post("/admin/documents/delete/{doc_id}")
def delete_document(request: Request, doc_id: int, db: Session = Depends(get_db)):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return RedirectResponse(url="/admin/documents")
    # delete file
    try:
        Path(doc.filepath).unlink(missing_ok=True)
    except Exception:
        pass
    # delete chunks
    db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == doc.id).delete()
    db.delete(doc)
    db.commit()
    return RedirectResponse(url="/admin/documents", status_code=303)
