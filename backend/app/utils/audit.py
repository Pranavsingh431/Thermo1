import logging
from typing import Optional, Dict, Any
from fastapi import Request
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)

# Cache DB capability for 'extra' column to avoid repeated metadata checks
_AUDIT_EXTRA_SUPPORTED: Optional[bool] = None


def write_audit_event(
    db: Session,
    *,
    user_id: Optional[int],
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    request: Optional[Request] = None,
    status_code: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    global _AUDIT_EXTRA_SUPPORTED
    try:
        ip = None
        ua = None
        method = None
        path = None
        if request is not None:
            ip = request.client.host if request.client else None
            ua = request.headers.get('User-Agent')
            method = request.method
            path = request.url.path
        # Detect support for 'extra' column once per process
        if _AUDIT_EXTRA_SUPPORTED is None:
            try:
                cols = db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='audit_logs'"))
                names = {r[0] for r in cols}
                _AUDIT_EXTRA_SUPPORTED = 'extra' in names
            except Exception:
                _AUDIT_EXTRA_SUPPORTED = False

        if not _AUDIT_EXTRA_SUPPORTED:
            # Skip audit write to avoid breaking requests on older DB schema
            return

        # Use a nested transaction to avoid poisoning the outer session
        with db.begin_nested():
            log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id is not None else None,
                method=method,
                path=path,
                status_code=status_code,
                ip_address=ip,
                user_agent=ua,
                extra=metadata or {},
            )
            db.add(log)
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass

