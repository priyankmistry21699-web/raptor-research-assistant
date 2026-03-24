"""
S3-compatible object storage client (MinIO / AWS S3).

Provides async-friendly wrappers around boto3 for document upload,
download, and deletion.
"""

import io
import logging
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3.endpoint,
        aws_access_key_id=settings.s3.access_key,
        aws_secret_access_key=settings.s3.secret_key,
        region_name=settings.s3.region,
    )


def upload_file(
    key: str, data: BinaryIO, content_type: str = "application/octet-stream"
) -> str:
    """Upload a file-like object and return the S3 key."""
    client = _get_client()
    client.upload_fileobj(
        data,
        settings.s3.bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    logger.info("Uploaded %s to bucket %s", key, settings.s3.bucket)
    return key


def download_file(key: str) -> bytes:
    """Download an object and return its bytes."""
    client = _get_client()
    buf = io.BytesIO()
    client.download_fileobj(settings.s3.bucket, key, buf)
    buf.seek(0)
    return buf.read()


def delete_file(key: str) -> None:
    """Delete a single object from the bucket."""
    client = _get_client()
    client.delete_object(Bucket=settings.s3.bucket, Key=key)
    logger.info("Deleted %s from bucket %s", key, settings.s3.bucket)


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned download URL."""
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3.bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def head_object(key: str) -> dict | None:
    """Return object metadata, or None if not found."""
    client = _get_client()
    try:
        return client.head_object(Bucket=settings.s3.bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise


def ensure_bucket() -> None:
    """Create the bucket if it doesn't exist (for dev/test)."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=settings.s3.bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3.bucket)
        logger.info("Created bucket %s", settings.s3.bucket)
