"""
Google Cloud Storage client.

Provides the same interface as s3_client but backed by GCS.
Used when STORAGE_PROVIDER=gcs.
"""

import io
import logging
from typing import BinaryIO

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        from google.cloud import storage as gcs_storage

        kwargs = {}
        if settings.storage.gcs_project:
            kwargs["project"] = settings.storage.gcs_project
        _client = gcs_storage.Client(**kwargs)
    return _client


def _get_bucket(bucket_name: str | None = None):
    client = _get_client()
    name = bucket_name or settings.storage.bucket
    return client.bucket(name)


def upload_file(
    key: str,
    data: BinaryIO,
    content_type: str = "application/octet-stream",
    bucket_name: str | None = None,
) -> str:
    """Upload a file-like object and return the key."""
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(key)
    blob.upload_from_file(data, content_type=content_type)
    logger.info("Uploaded %s to GCS bucket %s", key, bucket.name)
    return key


def download_file(key: str, bucket_name: str | None = None) -> bytes:
    """Download an object and return its bytes."""
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


def delete_file(key: str, bucket_name: str | None = None) -> None:
    """Delete a single object."""
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(key)
    blob.delete()
    logger.info("Deleted %s from GCS bucket %s", key, bucket.name)


def generate_presigned_url(key: str, expires_in: int = 3600, bucket_name: str | None = None) -> str:
    """Generate a signed download URL."""
    import datetime

    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(key)
    return blob.generate_signed_url(
        expiration=datetime.timedelta(seconds=expires_in),
        method="GET",
    )


def head_object(key: str, bucket_name: str | None = None) -> dict | None:
    """Return object metadata, or None if not found."""
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    blob.reload()
    return {
        "ContentLength": blob.size,
        "ContentType": blob.content_type,
        "LastModified": blob.updated,
    }


def ensure_bucket(bucket_name: str | None = None) -> None:
    """Create the bucket if it doesn't exist (for dev/test)."""
    client = _get_client()
    name = bucket_name or settings.storage.bucket
    bucket = client.bucket(name)
    if not bucket.exists():
        client.create_bucket(bucket, location="us-central1")
        logger.info("Created GCS bucket %s", name)


def list_blobs(prefix: str, bucket_name: str | None = None) -> list[str]:
    """List object keys under a prefix."""
    bucket = _get_bucket(bucket_name)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
