"""
RAPTOR RAG Platform — Main FastAPI Application

Factory pattern: ``create_app()`` builds the FastAPI instance so both
uvicorn (with ``--factory``) and tests can create isolated apps.

v1 routes remain mounted at their original paths for backward compat.
v2 routes live under ``/api/v2``.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Application factory — called once per process."""
    from app.core.logging_config import setup_logging
    setup_logging()

    app = FastAPI(
        title="RAPTOR RAG Platform",
        description="Production RAPTOR Retrieval-Augmented Generation platform",
        version="2.0.0-alpha",
    )

    # ── Middleware ─────────────────────────────────────────────────
    from app.core.middleware import register_middleware
    register_middleware(app)

    from app.core.security import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # ── Exception handlers ────────────────────────────────────────
    from app.core.exceptions import register_exception_handlers
    register_exception_handlers(app)

    # ── v1 routes (backward compatibility) ────────────────────────
    from app.api.retrieve import router as retrieve_router
    from app.api.train import router as train_router
    from app.api.chat import router as chat_router
    from app.api.feedback import router as feedback_router
    from app.api.eval import router as eval_router

    app.include_router(retrieve_router, prefix="/api/v1", tags=["v1"])
    app.include_router(train_router, prefix="/api/v1", tags=["v1"])
    app.include_router(chat_router, prefix="/api/v1", tags=["v1"])
    app.include_router(feedback_router, prefix="/api/v1", tags=["v1"])
    app.include_router(eval_router, prefix="/api/v1", tags=["v1"])

    # Also keep v1 routes at root for existing clients
    app.include_router(retrieve_router)
    app.include_router(train_router)
    app.include_router(chat_router)
    app.include_router(feedback_router)
    app.include_router(eval_router)

    # ── v2 routes ─────────────────────────────────────────────────
    from app.api.v2.routes.health import router as health_router
    from app.api.v2.routes.workspaces import router as ws_router
    from app.api.v2.routes.collections import router as coll_router
    from app.api.v2.routes.documents import router as doc_router
    from app.api.v2.routes.chat import router as chat_v2_router
    from app.api.v2.routes.retrieve import router as ret_v2_router
    from app.api.v2.routes.feedback import router as fb_v2_router
    from app.api.v2.routes.training import router as train_v2_router
    from app.api.v2.routes.auth import router as auth_router
    from app.api.v2.routes.generate import router as gen_router
    from app.api.v2.routes.eval import router as eval_v2_router
    from app.api.v2.routes.admin import router as admin_router

    v2_prefix = "/api/v2"
    app.include_router(health_router, prefix=v2_prefix)
    app.include_router(auth_router, prefix=v2_prefix)
    app.include_router(ws_router, prefix=v2_prefix)
    app.include_router(coll_router, prefix=v2_prefix)
    app.include_router(doc_router, prefix=v2_prefix)
    app.include_router(chat_v2_router, prefix=v2_prefix)
    app.include_router(ret_v2_router, prefix=v2_prefix)
    app.include_router(gen_router, prefix=v2_prefix)
    app.include_router(fb_v2_router, prefix=v2_prefix)
    app.include_router(eval_v2_router, prefix=v2_prefix)
    app.include_router(train_v2_router, prefix=v2_prefix)
    app.include_router(admin_router, prefix=v2_prefix)

    # ── Root info ─────────────────────────────────────────────────
    @app.get("/")
    def root():
        return {
            "service": "RAPTOR RAG Platform",
            "version": "2.0.0-alpha",
            "docs": "/docs",
            "health": "/api/v2/health/live",
        }

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "RAPTOR RAG Platform"}

    return app


# Backward-compatible module-level ``app`` for ``uvicorn app.main:app``
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=8000, reload=True)