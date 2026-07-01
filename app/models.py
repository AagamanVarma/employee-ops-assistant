from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy import func
from .database import Base


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    steps = Column(Text, nullable=False, default="")
    workflow_steps = relationship(
        "WorkflowStep",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="WorkflowStep.step_order",
    )


class WorkflowStep(Base):
    __tablename__ = "workflow_steps"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"), nullable=False)
    step_order = Column("order", Integer, nullable=False)
    description = Column(Text, nullable=False)

    workflow = relationship("Workflow", back_populates="workflow_steps")


class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(512), nullable=False)
    filepath = Column(String(1024), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    filename = Column(String(512), nullable=False)
    section_title = Column(String(512), nullable=True)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)

    document = relationship("Document", backref="chunks")
