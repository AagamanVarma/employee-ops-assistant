from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from jinja2 import Environment, FileSystemLoader
from starlette.middleware.sessions import SessionMiddleware
from app.routers import workflows, admin, documents, ask
from app.services.schema import ensure_schema
import os

app = FastAPI(title="Employee Ops Assistant")

# Session secret for admin login (override with ENV var in production)
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
# Disable Jinja2 template caching to avoid issues when request/session contains
# unhashable objects (keeps development simple and predictable).
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request},
    )

app.include_router(workflows.router, prefix="")
app.include_router(admin.router, prefix="")
app.include_router(documents.router, prefix="")
app.include_router(ask.router, prefix="")


@app.on_event("startup")
def startup_checks():
    ensure_schema()
