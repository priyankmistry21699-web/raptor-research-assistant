"""
Authentication middleware — Clerk JWT verification.

When CLERK_SECRET_KEY is not set (local dev), auth is bypassed and a
stub user context is injected.
"""

import logging
from typing import Optional

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Verify Clerk JWT from the Authorization header.
    Skips auth for public routes (health, docs, openapi).
    """

    PUBLIC_PREFIXES = ("/api/v2/health", "/docs", "/redoc", "/openapi.json", "/api/v1")

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes
        if any(request.url.path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return await call_next(request)

        # Dev mode bypass
        if not settings.auth.clerk_secret_key:
            request.state.user_id = None
            request.state.clerk_id = "dev-user"
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing authorization token")

        token = auth_header[7:]

        try:
            payload = _verify_clerk_token(token)
            request.state.clerk_id = payload.get("sub")
            request.state.user_id = None  # Resolved from DB in route dependencies
        except Exception:
            logger.warning("JWT verification failed for %s", request.url.path)
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return await call_next(request)


def _verify_clerk_token(token: str) -> dict:
    """
    Verify a Clerk-issued JWT.
    Uses PyJWT with Clerk's JWKS endpoint.
    """
    import jwt
    from jwt import PyJWKClient

    jwks_url = "https://api.clerk.com/.well-known/jwks.json"
    jwk_client = PyJWKClient(jwks_url)
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        options={"verify_aud": False},
    )
