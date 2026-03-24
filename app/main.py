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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Application factory — called once per process."""
    from app.core.logging_config import setup_logging
    from app.core.config import settings

    setup_logging()

    app = FastAPI(
        title="RAPTOR RAG Platform",
        description="Production RAPTOR Retrieval-Augmented Generation platform",
        version="2.0.0-alpha",
    )

    # ── Observability: Sentry ─────────────────────────────────────
    if settings.sentry_dsn:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=0.2 if settings.environment == "production" else 1.0,
            profiles_sample_rate=0.1,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            send_default_pii=False,
        )

    # ── Observability: OpenTelemetry ──────────────────────────────
    _init_telemetry(app, settings)

    # ── Middleware ─────────────────────────────────────────────────
    from app.core.middleware import register_middleware

    register_middleware(app)

    from app.core.security import AuthMiddleware

    app.add_middleware(AuthMiddleware)

    # ── Exception handlers ────────────────────────────────────────
    from app.core.exceptions import register_exception_handlers

    register_exception_handlers(app)

    # ── Observability: Prometheus ─────────────────────────────────
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            excluded_handlers=["/metrics", "/health", "/api/v2/health/live"],
        )
        instrumentator.instrument(app).expose(
            app, endpoint="/metrics", include_in_schema=False
        )
    except ImportError:
        pass  # prometheus-fastapi-instrumentator not installed

    # ── v1 routes (backward compatibility — deprecated) ───────────
    from app.api.retrieve import router as retrieve_router
    from app.api.train import router as train_router
    from app.api.chat import router as chat_router
    from app.api.feedback import router as feedback_router
    from app.api.eval import router as eval_router

    app.include_router(
        retrieve_router, prefix="/api/v1", tags=["v1-deprecated"], deprecated=True
    )
    app.include_router(
        train_router, prefix="/api/v1", tags=["v1-deprecated"], deprecated=True
    )
    app.include_router(
        chat_router, prefix="/api/v1", tags=["v1-deprecated"], deprecated=True
    )
    app.include_router(
        feedback_router, prefix="/api/v1", tags=["v1-deprecated"], deprecated=True
    )
    app.include_router(
        eval_router, prefix="/api/v1", tags=["v1-deprecated"], deprecated=True
    )

    # Also keep v1 routes at root for existing clients
    app.include_router(retrieve_router, deprecated=True)
    app.include_router(train_router, deprecated=True)
    app.include_router(chat_router, deprecated=True)
    app.include_router(feedback_router, deprecated=True)
    app.include_router(eval_router, deprecated=True)

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

    # ── Startup validation ────────────────────────────────────────
    @app.on_event("startup")
    async def _startup_checks():
        if settings.environment == "production":
            assert settings.secret_key != "change-me-in-production", (
                "SECRET_KEY must be changed in production"
            )
            assert settings.auth.clerk_secret_key, (
                "CLERK_SECRET_KEY is required in production"
            )

    return app


def _init_telemetry(app, settings):
    """Initialize OpenTelemetry tracing if OTLP endpoint is configured."""
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create(
            {
                "service.name": "raptor-rag-platform",
                "service.version": "2.0.0-alpha",
                "deployment.environment": settings.environment,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass  # OTel packages not installed


# Backward-compatible module-level ``app`` for ``uvicorn app.main:app``
app = create_app()

if __name__ == "__main__":
    import uvicorn

    from app.core.config import settings

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
