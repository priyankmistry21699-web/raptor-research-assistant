"""
RAPTOR RAG Platform — Application Settings

Centralized configuration loaded from environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation and .env file support.
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


class S3Settings(BaseSettings):
    endpoint: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT")
    bucket: str = Field(default="raptor-documents", alias="S3_BUCKET")
    access_key: str = Field(default="minioadmin", alias="AWS_ACCESS_KEY_ID")
    secret_key: str = Field(default="minioadmin", alias="AWS_SECRET_ACCESS_KEY")
    region: str = Field(default="us-east-1", alias="S3_REGION")


class LLMSettings(BaseSettings):
    provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    model: str = Field(default="mistral:latest", alias="LLM_MODEL")
    ollama_base_url: str = Field(
        default="http://localhost:11435", alias="OLLAMA_BASE_URL"
    )
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")


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
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
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
    s3: S3Settings = S3Settings()
    llm: LLMSettings = LLMSettings()
    auth: AuthSettings = AuthSettings()

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # RAPTOR
    raptor_max_topics: int = 5
    raptor_chunk_size: int = 500

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton instance
settings = Settings()
