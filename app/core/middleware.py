"""
Middleware stack for the FastAPI application.

Includes:
  - Request ID injection
  - Request timing / structured logging
  - CORS (configured from settings)
  - Security headers
  - Rate limiting (Redis-backed)
"""

import time
import uuid
import logging

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger("raptor.access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject a unique request-id header into every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if settings.environment == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-backed per-user sliding-window rate limiter.
    Limits authenticated users by clerk_id, anonymous by IP.
    """

    # Routes with stricter limits
    STRICT_PREFIXES = ("/api/v2/generate",)
    DEFAULT_MAX = 120       # requests per window
    DEFAULT_WINDOW = 60     # seconds
    STRICT_MAX = 20         # stricter for generation
    STRICT_WINDOW = 60

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health/docs
        path = request.url.path
        if path.startswith(("/api/v2/health", "/docs", "/redoc", "/openapi.json", "/health")):
            return await call_next(request)

        # Determine identity key
        identity = getattr(request.state, "clerk_id", None)
        if not identity:
            identity = request.client.host if request.client else "unknown"

        # Choose limits
        is_strict = any(path.startswith(p) for p in self.STRICT_PREFIXES)
        max_req = self.STRICT_MAX if is_strict else self.DEFAULT_MAX
        window = self.STRICT_WINDOW if is_strict else self.DEFAULT_WINDOW

        rl_key = f"rl:{identity}:{path.split('/')[3] if len(path.split('/')) > 3 else 'default'}"

        try:
            from app.storage.cache import check_rate_limit
            allowed = check_rate_limit(rl_key, max_req, window)
        except Exception:
            # If Redis is down, allow the request (fail-open)
            allowed = True

        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(window)},
            )

        return await call_next(request)


def register_middleware(app: FastAPI) -> None:
    """Register all middleware in the correct order (outermost first)."""

    # CORS — must be outermost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    app.add_middleware(RateLimitMiddleware)

    # Request ID
    app.add_middleware(RequestIdMiddleware)

    # Timing / access logs
    app.add_middleware(TimingMiddleware)
