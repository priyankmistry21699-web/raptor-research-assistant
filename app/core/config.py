"""
RAPTOR RAG Platform — Application Settings

Centralized configuration loaded from environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation and .env file support.

Supports both local dev (MinIO, local Redis) and GCP production (GCS, Cloud SQL,
Memorystore) via the same env-var interface.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class DatabaseSettings(BaseSettings):
    url: str = Field(
        default="postgresql+asyncpg://raptor:raptor@localhost:5432/raptor",
        alias="DATABASE_URL",
    )
    url_sync: str = Field(
        default="postgresql://raptor:raptor@localhost:5432/raptor",
        alias="DATABASE_URL_SYNC",
    )
    pool_size: int = 20
    max_overflow: int = 10


class RedisSettings(BaseSettings):
    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class QdrantSettings(BaseSettings):
    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    collection_prefix: str = "raptor"


class StorageSettings(BaseSettings):
    """Object storage — works with both MinIO (local) and GCS (production)."""

    provider: str = Field(default="s3", alias="STORAGE_PROVIDER")  # "s3" or "gcs"
    # S3 / MinIO settings (local dev default)
    endpoint: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT")
    bucket: str = Field(default="raptor-documents", alias="STORAGE_BUCKET")
    access_key: str = Field(default="minioadmin", alias="AWS_ACCESS_KEY_ID")
    secret_key: str = Field(default="minioadmin", alias="AWS_SECRET_ACCESS_KEY")
    region: str = Field(default="us-east-1", alias="S3_REGION")
    # GCS settings (production)
    gcs_project: Optional[str] = Field(default=None, alias="GCS_PROJECT")
    gcs_credentials_json: Optional[str] = Field(
        default=None, alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    # Additional buckets
    tree_bucket: str = Field(default="raptor-trees", alias="STORAGE_TREE_BUCKET")
    model_bucket: str = Field(default="raptor-models", alias="STORAGE_MODEL_BUCKET")


class LLMSettings(BaseSettings):
    provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    model: str = Field(default="mistral:latest", alias="LLM_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11435", alias="OLLAMA_BASE_URL"
    )
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")


class RerankerSettings(BaseSettings):
    enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    model: str = Field(default="BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    top_k: int = Field(default=5, alias="RERANKER_TOP_K")


class AuthSettings(BaseSettings):
    clerk_secret_key: Optional[str] = Field(default=None, alias="CLERK_SECRET_KEY")
    clerk_publishable_key: Optional[str] = Field(
        default=None, alias="CLERK_PUBLISHABLE_KEY"
    )
    clerk_webhook_secret: Optional[str] = Field(
        default=None, alias="CLERK_WEBHOOK_SECRET"
    )


class Settings(BaseSettings):
    """Root application settings."""

    # App
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # Server
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    frontend_url: str = Field(default="http://localhost:3000", alias="FRONTEND_URL")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:7860"],
        alias="CORS_ORIGINS",
    )

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")

    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    qdrant: QdrantSettings = QdrantSettings()
    storage: StorageSettings = StorageSettings()
    llm: LLMSettings = LLMSettings()
    reranker: RerankerSettings = RerankerSettings()
    auth: AuthSettings = AuthSettings()

    # Backward compat alias
    @property
    def s3(self) -> StorageSettings:
        return self.storage

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RAPTOR
    raptor_max_topics: int = 5
    raptor_chunk_size: int = 500
    raptor_chunk_overlap: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance
settings = Settings()
