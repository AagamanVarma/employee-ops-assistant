from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from app import models
from app.database import get_db, engine
from app.routers.admin import get_current_admin

models.Base.metadata.create_all(bind=engine)

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env = Environment(loader=FileSystemLoader("templates"), cache_size=0)


def _build_steps(steps_text: str):
    """Create clean workflow step text list from submitted textarea."""
    return [s.strip() for s in steps_text.splitlines() if s.strip()]


@router.get("/workflows")
def list_workflows(request: Request, db: Session = Depends(get_db)):
    items = db.query(models.Workflow).all()
    return templates.TemplateResponse(
        request,
        "workflows/list.html",
        {"request": request, "workflows": items, "is_admin": bool(get_current_admin(request))},
    )


@router.get("/workflows/create")
def create_workflow_form(request: Request):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    return templates.TemplateResponse(
        request,
        "workflows/create.html",
        {"request": request, "is_admin": True},
    )


@router.post("/workflows/create")
def create_workflow(request: Request, name: str = Form(...), description: str = Form(None), steps: str = Form(...), db: Session = Depends(get_db)):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    steps_list = _build_steps(steps)
    workflow_name = name.strip()
    wf = db.query(models.Workflow).filter(models.Workflow.name == workflow_name).first()
    if wf:
        wf.description = description
        wf.steps = "\n".join(steps_list)
        wf.workflow_steps.clear()
        db.flush()
    else:
        wf = models.Workflow(
            name=workflow_name,
            description=description,
            steps="\n".join(steps_list),
        )
        db.add(wf)
        db.flush()

    for index, step_text in enumerate(steps_list, start=1):
        wf_step = models.WorkflowStep(
            workflow_id=wf.id,
            step_order=index,
            description=step_text,
        )
        db.add(wf_step)

    db.commit()
    return RedirectResponse(url="/workflows", status_code=303)


@router.get("/workflows/{workflow_id}")
def view_workflow(workflow_id: int, request: Request, db: Session = Depends(get_db)):
    wf = db.query(models.Workflow).filter(models.Workflow.id == workflow_id).first()
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    steps = [step.description for step in sorted(wf.workflow_steps, key=lambda s: s.step_order)]
    return templates.TemplateResponse(
        request,
        "workflows/view.html",
        {"request": request, "workflow": wf, "steps": steps, "is_admin": bool(get_current_admin(request))},
    )


@router.post("/workflows/{workflow_id}/delete")
def delete_workflow(request: Request, workflow_id: int, db: Session = Depends(get_db)):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    wf = db.query(models.Workflow).filter(models.Workflow.id == workflow_id).first()
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    db.delete(wf)
    db.commit()
    return RedirectResponse(url="/workflows", status_code=303)
