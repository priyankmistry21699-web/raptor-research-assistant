"""
Authentication & Authorization — Clerk JWT verification + RBAC.

Security model:
  - Clerk JWT verification for all protected routes
  - Dev mode: explicit opt-in via ENVIRONMENT=development (viewer role only)
  - Role-based access control: admin, editor, viewer
  - Route-level dependencies: get_current_user, require_role, require_roles
"""

import logging

from fastapi import Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Role definitions ──────────────────────────────────────────────────

ROLES = ("admin", "editor", "viewer")
ROLE_HIERARCHY = {"admin": 3, "editor": 2, "viewer": 1}


# ── Auth middleware ───────────────────────────────────────────────────


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Verify Clerk JWT from the Authorization header.
    Skips auth for public routes (health, docs, openapi, webhook).
    """

    PUBLIC_PREFIXES = (
        "/api/v2/health",
        "/api/v2/auth/webhook",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/",
    )

    PUBLIC_EXACT = ("/", "/health")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for exact public paths
        if path in self.PUBLIC_EXACT:
            return await call_next(request)

        # Skip auth for public prefixes
        if any(
            path.startswith(p)
            for p in self.PUBLIC_PREFIXES
            if p not in self.PUBLIC_EXACT
        ):
            return await call_next(request)

        # Dev mode bypass — only when environment is explicitly "development"
        if settings.environment == "development" and not settings.auth.clerk_secret_key:
            request.state.user_id = None
            request.state.clerk_id = "dev-user"
            request.state.user_role = "viewer"  # Never grant admin in dev bypass
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
            request.state.user_role = payload.get("role", "viewer")
        except Exception:
            logger.warning("JWT verification failed for %s", path)
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return await call_next(request)


# ── JWT verification ──────────────────────────────────────────────────


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


# ── Route-level dependencies ─────────────────────────────────────────


def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency — extract the authenticated user from request state.
    Returns a dict with user_id, clerk_id, and role.
    Raises 401 if not authenticated.
    """
    clerk_id = getattr(request.state, "clerk_id", None)
    if not clerk_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {
        "user_id": getattr(request.state, "user_id", None),
        "clerk_id": clerk_id,
        "role": getattr(request.state, "user_role", "viewer"),
    }


def require_role(role: str):
    """
    FastAPI dependency factory — enforce minimum role level.
    Usage: Depends(require_role("admin"))
    """
    min_level = ROLE_HIERARCHY.get(role, 0)

    def _check(current_user: dict = Depends(get_current_user)) -> dict:
        user_level = ROLE_HIERARCHY.get(current_user.get("role", ""), 0)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {role} role or higher",
            )
        return current_user

    return _check


def require_roles(*roles: str):
    """
    FastAPI dependency factory — enforce that user has one of the listed roles.
    Usage: Depends(require_roles("admin", "editor"))
    """

    def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires one of: {', '.join(roles)}",
            )
        return current_user

    return _check
