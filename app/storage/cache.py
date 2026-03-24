"""
Redis cache client.

Provides simple get/set/delete with optional TTL, plus lightweight
rate-limiting helpers.
"""

import json
import logging
from typing import Any

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: redis.ConnectionPool | None = None


def _get_client() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(settings.redis.url, decode_responses=True)
    return redis.Redis(connection_pool=_pool)


# ── Key-value cache ───────────────────────────────────────────────────


def cache_get(key: str) -> Any | None:
    """Return the cached value (JSON-decoded), or None."""
    client = _get_client()
    raw = client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Store a JSON-serializable value with a TTL."""
    client = _get_client()
    client.setex(key, ttl_seconds, json.dumps(value))


def cache_delete(key: str) -> None:
    client = _get_client()
    client.delete(key)


def cache_delete_pattern(pattern: str) -> int:
    """Delete all keys matching a glob pattern. Returns count deleted."""
    client = _get_client()
    keys = list(client.scan_iter(match=pattern, count=500))
    if keys:
        client.delete(*keys)
    return len(keys)


# ── Rate limiting ─────────────────────────────────────────────────────


def check_rate_limit(key: str, max_requests: int, window_seconds: int) -> bool:
    """
    Sliding-window counter rate limit.
    Returns True if request is allowed, False if rate-limited.
    """
    client = _get_client()
    pipe = client.pipeline(transaction=True)
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    results = pipe.execute()
    current = results[0]
    return current <= max_requests


# ── Health ────────────────────────────────────────────────────────────


def ping() -> bool:
    """Return True if Redis is reachable."""
    try:
        client = _get_client()
        return client.ping()
    except Exception:
        return False
