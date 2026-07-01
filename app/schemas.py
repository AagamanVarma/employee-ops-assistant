from pydantic import BaseModel
from typing import List, Optional


class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    steps: List[str]


class WorkflowRead(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    steps: List[str]

    class Config:
        from_attributes = True
