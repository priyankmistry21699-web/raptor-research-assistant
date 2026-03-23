"""
Audit logging utility.

Call ``log_audit()`` from route handlers and middleware to record
user actions as immutable audit trail entries in the database.
"""

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.db.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


def log_audit(
    db: Session,
    *,
    user_id: uuid.UUID | None = None,
    action: str,
    resource: str | None = None,
    resource_id: uuid.UUID | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> AuditLog:
    """
    Write an audit log entry.

    Args:
        db: Active SQLAlchemy session.
        user_id: Authenticated user UUID (optional).
        action: Short action verb (e.g. "document.upload", "collection.delete").
        resource: Resource type (e.g. "document", "collection", "model").
        resource_id: UUID of the affected resource.
        details: Arbitrary JSON metadata.
        ip_address: Client IP address.

    Returns:
        The persisted AuditLog row.
    """
    entry = AuditLog(
        id=uuid.uuid4(),
        user_id=user_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
    )
    db.add(entry)
    # Do not commit — let the caller's transaction boundary handle it
    logger.debug("Audit: %s %s/%s by user %s", action, resource, resource_id, user_id)
    return entry


def log_audit_from_request(
    db: Session,
    request,
    *,
    action: str,
    resource: str | None = None,
    resource_id: uuid.UUID | None = None,
    details: dict[str, Any] | None = None,
) -> AuditLog:
    """
    Convenience wrapper that extracts user_id and IP from a FastAPI Request.
    """
    user_id = getattr(request.state, "user_id", None)
    ip = request.client.host if request.client else None
    return log_audit(
        db,
        user_id=user_id,
        action=action,
        resource=resource,
        resource_id=resource_id,
        details=details,
        ip_address=ip,
    )
