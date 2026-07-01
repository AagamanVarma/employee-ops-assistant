from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from app.services.retrieval import retrieve

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)

EXAMPLE_QUERIES = [
    "How do I regularize attendance?",
    "How many sick leaves can interns take?",
    "How do I fill timesheet?",
    "What is the leave policy for interns?",
]


@router.get("/ask")
def ask_form(request: Request):
    return templates.TemplateResponse(
        request,
        "ask/search.html",
        {"request": request, "query": "", "examples": EXAMPLE_QUERIES, "results": None},
    )


@router.post("/ask")
def ask_search(request: Request, query: str = Form(...)):
    results = retrieve(query, top_k=5)
    found_policy = len(results.get("chunks", [])) > 0
    found_workflows = len(results.get("workflows", [])) > 0
    show_fallback = not found_policy and not found_workflows
    return templates.TemplateResponse(
        request,
        "ask/results.html",
        {
            "request": request,
            "query": query,
            "examples": EXAMPLE_QUERIES,
            "results": results,
            "show_fallback": show_fallback,
            "debug": results.get("debug", {}),
            "contact_hr": show_fallback,
        },
    )
