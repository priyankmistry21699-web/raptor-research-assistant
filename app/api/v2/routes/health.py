"""
Health & readiness endpoints — /api/v2/health
"""

from fastapi import APIRouter

from app.api.v2.schemas import HealthCheck, ReadinessCheck
from app.storage.cache import ping as redis_ping

router = APIRouter(prefix="/health", tags=["health"])

APP_VERSION = "2.0.0-alpha"


@router.get("/live", response_model=HealthCheck)
async def liveness():
    """Kubernetes / Docker liveness probe."""
    return HealthCheck(status="ok", version=APP_VERSION)


@router.get("/ready", response_model=ReadinessCheck)
async def readiness():
    """Readiness probe — checks all backing services."""
    from app.db.session import async_engine
    from sqlalchemy import text as sa_text

    db_ok = "ok"
    try:
        async with async_engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
    except Exception:
        db_ok = "unavailable"

    redis_ok = "ok" if redis_ping() else "unavailable"

    # Qdrant
    qdrant_ok = "ok"
    try:
        from qdrant_client import QdrantClient
        from app.core.config import settings

        client = QdrantClient(
            url=settings.qdrant.url, api_key=settings.qdrant.api_key, timeout=3
        )
        client.get_collections()
    except Exception:
        qdrant_ok = "unavailable"

    # S3
    s3_ok = "ok"
    try:
        from app.storage.s3_client import _get_client
        from app.core.config import settings as s

        _get_client().head_bucket(Bucket=s.s3.bucket)
    except Exception:
        s3_ok = "unavailable"

    overall = (
        "ok"
        if all(s == "ok" for s in [db_ok, redis_ok, qdrant_ok, s3_ok])
        else "degraded"
    )
    return ReadinessCheck(
        status=overall, database=db_ok, redis=redis_ok, qdrant=qdrant_ok, s3=s3_ok
    )
