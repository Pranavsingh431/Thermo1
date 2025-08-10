import io
import json
import os
import time
from datetime import timedelta
from typing import Optional

from app.config import settings


class StorageClient:
    def __init__(self):
        self.provider = settings.OBJECT_STORAGE_PROVIDER.lower()
        self.bucket = settings.OBJECT_STORAGE_BUCKET
        self._s3 = None
        self._gcs = None

    def upload_file(self, local_path: str, remote_key: str) -> str:
        if not settings.USE_OBJECT_STORAGE or self.provider == "none":
            return local_path
        if self.provider == "s3":
            import boto3
            if self._s3 is None:
                self._s3 = boto3.client(
                    "s3",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
            self._s3.upload_file(local_path, self.bucket, remote_key)
            return f"s3://{self.bucket}/{remote_key}"
        elif self.provider == "gcs":
            from google.cloud import storage  # type: ignore
            if self._gcs is None:
                if settings.GCS_SERVICE_ACCOUNT_JSON and os.path.exists(settings.GCS_SERVICE_ACCOUNT_JSON):
                    self._gcs = storage.Client.from_service_account_json(settings.GCS_SERVICE_ACCOUNT_JSON)
                else:
                    self._gcs = storage.Client()
            bucket = self._gcs.bucket(self.bucket)
            blob = bucket.blob(remote_key)
            blob.upload_from_filename(local_path)
            return f"gs://{self.bucket}/{remote_key}"
        else:
            return local_path

    def presign_download(self, remote_key: str, expires_seconds: int = 900) -> Optional[str]:
        if not settings.USE_OBJECT_STORAGE or self.provider == "none":
            return None
        if self.provider == "s3":
            import boto3
            if self._s3 is None:
                self._s3 = boto3.client(
                    "s3",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": remote_key},
                ExpiresIn=expires_seconds,
            )
        elif self.provider == "gcs":
            from google.cloud import storage  # type: ignore
            if self._gcs is None:
                if settings.GCS_SERVICE_ACCOUNT_JSON and os.path.exists(settings.GCS_SERVICE_ACCOUNT_JSON):
                    self._gcs = storage.Client.from_service_account_json(settings.GCS_SERVICE_ACCOUNT_JSON)
                else:
                    self._gcs = storage.Client()
            bucket = self._gcs.bucket(self.bucket)
            blob = bucket.blob(remote_key)
            return blob.generate_signed_url(expiration=timedelta(seconds=expires_seconds), method="GET")
        return None

    def presign_upload(self, remote_key: str, content_type: str = "application/octet-stream", expires_seconds: int = 900) -> Optional[dict]:
        """Return dict with url and headers/fields for client-side upload."""
        if not settings.USE_OBJECT_STORAGE or self.provider == "none":
            return None
        if self.provider == "s3":
            import boto3
            if self._s3 is None:
                self._s3 = boto3.client(
                    "s3",
                    region_name=settings.AWS_REGION,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                )
            # Use PUT presign for simplicity
            url = self._s3.generate_presigned_url(
                ClientMethod='put_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': remote_key,
                    'ContentType': content_type,
                },
                ExpiresIn=expires_seconds
            )
            return {"provider": "s3", "url": url, "headers": {"Content-Type": content_type}, "key": remote_key}
        elif self.provider == "gcs":
            from google.cloud import storage  # type: ignore
            if self._gcs is None:
                if settings.GCS_SERVICE_ACCOUNT_JSON and os.path.exists(settings.GCS_SERVICE_ACCOUNT_JSON):
                    self._gcs = storage.Client.from_service_account_json(settings.GCS_SERVICE_ACCOUNT_JSON)
                else:
                    self._gcs = storage.Client()
            bucket = self._gcs.bucket(self.bucket)
            blob = bucket.blob(remote_key)
            url = blob.generate_signed_url(expiration=timedelta(seconds=expires_seconds), method="PUT", content_type=content_type)
            return {"provider": "gcs", "url": url, "headers": {"Content-Type": content_type}, "key": remote_key}
        return None


storage_client = StorageClient()

