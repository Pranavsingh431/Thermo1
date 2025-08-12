from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import secrets

from app.database import Base

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    token = Column(String(255), unique=True, index=True, nullable=False)
    family_id = Column(String(255), index=True, nullable=False)  # For token family rotation
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="refresh_tokens")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    device_info = Column(String(500))  # User agent, IP, etc.
    last_used = Column(DateTime(timezone=True))
    
    @classmethod
    def generate_token(cls) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(32)
    
    @classmethod
    def generate_family_id(cls) -> str:
        """Generate a unique family ID for token rotation"""
        return secrets.token_urlsafe(16)
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)"""
        return not self.is_expired() and not self.is_revoked
    
    def revoke(self) -> None:
        """Revoke this token"""
        self.is_revoked = True
    
    def __repr__(self):
        return f"<RefreshToken(id={self.id}, user_id={self.user_id}, family_id='{self.family_id}', expired={self.is_expired()})>"
