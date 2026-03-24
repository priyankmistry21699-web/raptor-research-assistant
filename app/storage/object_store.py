"""
Unified object storage interface.

Routes calls to S3/MinIO or GCS based on ``settings.storage.provider``.
All application code should import from this module instead of s3_client or gcs_client directly.
"""

from __future__ import annotations

from typing import BinaryIO

from app.core.config import settings


def _backend():
    if settings.storage.provider == "gcs":
        from app.storage import gcs_client as mod
    else:
        from app.storage import s3_client as mod
    return mod


def upload_file(
    key: str,
    data: BinaryIO,
    content_type: str = "application/octet-stream",
    bucket_name: str | None = None,
) -> str:
    mod = _backend()
    if settings.storage.provider == "gcs":
        return mod.upload_file(
            key, data, content_type=content_type, bucket_name=bucket_name
        )
    return mod.upload_file(key, data, content_type=content_type)


def download_file(key: str, bucket_name: str | None = None) -> bytes:
    mod = _backend()
    if settings.storage.provider == "gcs" and bucket_name:
        return mod.download_file(key, bucket_name=bucket_name)
    return mod.download_file(key)


def delete_file(key: str, bucket_name: str | None = None) -> None:
    mod = _backend()
    if settings.storage.provider == "gcs" and bucket_name:
        mod.delete_file(key, bucket_name=bucket_name)
    else:
        mod.delete_file(key)


def generate_presigned_url(
    key: str, expires_in: int = 3600, bucket_name: str | None = None
) -> str:
    mod = _backend()
    if settings.storage.provider == "gcs" and bucket_name:
        return mod.generate_presigned_url(
            key, expires_in=expires_in, bucket_name=bucket_name
        )
    return mod.generate_presigned_url(key, expires_in=expires_in)


def head_object(key: str, bucket_name: str | None = None) -> dict | None:
    mod = _backend()
    if settings.storage.provider == "gcs" and bucket_name:
        return mod.head_object(key, bucket_name=bucket_name)
    return mod.head_object(key)


def ensure_bucket(bucket_name: str | None = None) -> None:
    mod = _backend()
    if settings.storage.provider == "gcs":
        mod.ensure_bucket(bucket_name=bucket_name)
    else:
        mod.ensure_bucket()
