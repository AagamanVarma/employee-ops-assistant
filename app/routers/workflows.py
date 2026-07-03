from fastapi import APIRouter, Depends, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
import logging
from app import models
from app.database import get_db, engine
from app.routers.admin import get_current_admin
from app.services.embeddings import emb_service

logger = logging.getLogger(__name__)

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
        if not wf.vector_id:
            wf.vector_id = f"workflow_{wf.id}"
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
        wf.vector_id = f"workflow_{wf.id}"

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
def delete_workflow(
    request: Request,
    workflow_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if not get_current_admin(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    wf = db.query(models.Workflow).filter(models.Workflow.id == workflow_id).first()
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    vector_ids = [wf.vector_id] if wf.vector_id else []

    try:
        db.delete(wf)
        db.commit()
        logger.debug("Deleted workflow row %s and cascade-deleted its steps", workflow_id)
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete workflow %s: %s", workflow_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete workflow")

    def _cleanup_workflow_vectors(ids: list[str]):
        try:
            result = emb_service.delete_vector_ids(ids)
            logger.debug("Background Chroma cleanup finished for workflow %s ids=%s: %s", workflow_id, ids, result)
        except Exception as exc:
            logger.warning(
                "Background Chroma cleanup failed for workflow %s ids=%s: %s",
                workflow_id,
                ids,
                exc,
                exc_info=True,
            )

    background_tasks.add_task(_cleanup_workflow_vectors, vector_ids)
    logger.debug("Scheduled Chroma cleanup background task for workflow %s", workflow_id)

    return RedirectResponse(url="/workflows", status_code=303)
