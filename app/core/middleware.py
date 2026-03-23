"""
Middleware stack for the FastAPI application.

Includes:
  - Request ID injection
  - Request timing
  - Structured logging
  - CORS (configured from settings)
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


def register_middleware(app: FastAPI) -> None:
    """Register all middleware in the correct order (outermost first)."""

    # CORS — must be outermost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID
    app.add_middleware(RequestIdMiddleware)

    # Timing / access logs
    app.add_middleware(TimingMiddleware)
