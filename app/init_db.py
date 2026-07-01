from app.database import engine
from app import models
from app.services.schema import ensure_schema


def init_db():
    """Create all database tables for the app."""
    models.Base.metadata.create_all(bind=engine)
    ensure_schema()
