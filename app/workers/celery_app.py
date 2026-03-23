"""
Celery application factory.

Uses Redis as both broker and result backend.
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "raptor",
    broker=settings.redis.url,
    backend=settings.redis.url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,  # 24 hours
)

# Auto-discover tasks in app.workers.tasks
celery_app.autodiscover_tasks(["app.workers.tasks"])
