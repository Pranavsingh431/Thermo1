from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class TaskRun(Base):
    __tablename__ = "task_runs"

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String(200), index=True, nullable=False)
    task_id = Column(String(100), index=True, nullable=False)
    args = Column(JSON)
    kwargs = Column(JSON)
    status = Column(String(50), index=True, nullable=False, default="started")
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    error_message = Column(Text)
    worker_hostname = Column(String(200))

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

