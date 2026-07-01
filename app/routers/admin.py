from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app import models
from app.utils import hash_password, verify_password
import json

models.Base.metadata.create_all(bind=engine)

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)


def get_current_admin(request: Request):
    admin = request.session.get("admin")
    return admin


@router.get("/admin/login")
def login_form(request: Request):
    return templates.TemplateResponse(request, "admin/login.html", {"request": request})


@router.post("/admin/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.Admin).filter(models.Admin.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"request": request, "error": "Invalid credentials"},
        )
    request.session["admin"] = user.username
    return RedirectResponse(url="/admin", status_code=303)


@router.get("/admin/logout")
def logout(request: Request):
    request.session.pop("admin", None)
    return RedirectResponse(url="/", status_code=303)


@router.get("/admin")
def admin_index(request: Request, db: Session = Depends(get_db)):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login")
    workflows = db.query(models.Workflow).all()
    return templates.TemplateResponse(
        request,
        "admin/index.html",
        {"request": request, "workflows": workflows},
    )


@router.get("/admin/setup")
def admin_setup(request: Request, db: Session = Depends(get_db)):
    # Create default admin if none exists
    existing = db.query(models.Admin).first()
    if existing:
        return RedirectResponse(url="/admin", status_code=303)
    # default credentials: admin / admin (user should change)
    admin = models.Admin(username="admin", password_hash=hash_password("admin"))
    db.add(admin)
    db.commit()
    return RedirectResponse(url="/admin/login", status_code=303)
