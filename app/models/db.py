from sqlmodel import SQLModel, Field, create_engine, Session
from datetime import datetime
from ..core.config import settings

engine = create_engine(f"sqlite:///{settings.db_path}", echo=False)

class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    status: str = "queued"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message: str | None = None
    index_name: str | None = None
    source_filename: str | None = None

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
