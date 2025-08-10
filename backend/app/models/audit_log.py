from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func

from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True, index=True)
    method = Column(String(10), nullable=True)
    path = Column(String(500), nullable=True)
    status_code = Column(Integer, nullable=True)
    ip_address = Column(String(100), nullable=True)
    user_agent = Column(String(300), nullable=True)
    extra = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action}, resource={self.resource_type}:{self.resource_id})>"

