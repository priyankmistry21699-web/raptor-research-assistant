"""
Auth routes — Clerk webhook + user info.

POST /auth/webhook  — Clerk user.created / user.updated / user.deleted events
GET  /auth/me       — Current authenticated user info
"""

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session

from app.api.v2.schemas import UserOut, Message
from app.db.session import get_db_sync
from app.db.models.user import User
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _verify_webhook_signature(payload: bytes, signature: str | None) -> bool:
    """Verify the Clerk webhook signature (svix)."""
    secret = settings.auth.clerk_webhook_secret
    if not secret:
        # Dev mode — skip verification
        return True
    if not signature:
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhook", response_model=Message)
async def clerk_webhook(request: Request, db: Session = Depends(get_db_sync)):
    """
    Handle Clerk webhook events for user sync.
    Supported events: user.created, user.updated, user.deleted.
    """
    body = await request.body()
    sig = request.headers.get("svix-signature")

    if not _verify_webhook_signature(body, sig):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    import json
    event = json.loads(body)
    event_type = event.get("type", "")
    data = event.get("data", {})

    clerk_id = data.get("id")
    if not clerk_id:
        raise HTTPException(status_code=400, detail="Missing user id in event data")

    if event_type == "user.created":
        email = _extract_email(data)
        display_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip() or None
        user = User(
            id=uuid.uuid4(),
            clerk_id=clerk_id,
            email=email,
            display_name=display_name,
            role="member",
        )
        db.add(user)
        db.commit()
        logger.info("Created user %s from Clerk webhook", clerk_id)

    elif event_type == "user.updated":
        user = db.query(User).filter(User.clerk_id == clerk_id).first()
        if user:
            email = _extract_email(data)
            if email:
                user.email = email
            display_name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            if display_name:
                user.display_name = display_name
            db.commit()

    elif event_type == "user.deleted":
        user = db.query(User).filter(User.clerk_id == clerk_id).first()
        if user:
            db.delete(user)
            db.commit()
            logger.info("Deleted user %s from Clerk webhook", clerk_id)

    return Message(detail="ok")


def _extract_email(data: dict) -> str:
    """Extract primary email from Clerk event data."""
    addresses = data.get("email_addresses", [])
    for addr in addresses:
        if addr.get("id") == data.get("primary_email_address_id"):
            return addr["email_address"]
    if addresses:
        return addresses[0]["email_address"]
    return data.get("email", "unknown@example.com")


@router.get("/me", response_model=UserOut)
def get_current_user(request: Request, db: Session = Depends(get_db_sync)):
    """Return the current authenticated user's info."""
    clerk_id = getattr(request.state, "clerk_id", None)
    if not clerk_id or clerk_id == "dev-user":
        # Dev mode — return a stub
        return UserOut(
            id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
            email="dev@localhost",
            display_name="Dev User",
            role="admin",
            created_at=datetime.now(timezone.utc),
        )

    user = db.query(User).filter(User.clerk_id == clerk_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
