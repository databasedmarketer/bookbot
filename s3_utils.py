"""
Helper utilities for reading book data (CSV + TXT) from S3.
"""
import boto3
from typing import List, Tuple
from botocore.exceptions import ClientError
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BEDROCK_REGION


def get_s3_client():
    client_kwargs = {"region_name": BEDROCK_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        client_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **client_kwargs)

def fetch_csv_bytes(bucket: str, key: str) -> bytes:
    """Download a single CSV object from S3 and return its raw bytes."""
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        raise IOError(
            f"Failed to download s3://{bucket}/{key}: {e}\n"
            f"Please check the bucket name, key, and AWS credentials/permissions."
        )


def list_txt_keys(bucket: str, prefix: str) -> List[str]:
    """List all .txt object keys under a given S3 prefix (folder)."""
    s3 = get_s3_client()
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".txt"):
                keys.append(obj["Key"])
    return keys


def fetch_txt_files(bucket: str, prefix: str) -> List[Tuple[str, str]]:
    """
    Download every .txt file under prefix.
    Returns (filename, text_content) tuples — filename is just the base
    name (no folder path) so the existing fuzzy title-matching logic in
    BookRAGPipeline._match_txt keeps working unchanged.
    """
    s3 = get_s3_client()
    results: List[Tuple[str, str]] = []
    for key in list_txt_keys(bucket, prefix):
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8", errors="replace")
        filename = key.split("/")[-1]
        results.append((filename, content))
    return results


def list_keys_with_suffix(bucket: str, prefix: str, suffix: str) -> List[str]:
    """List all S3 object keys under `prefix` whose key ends with `suffix` (case-insensitive)."""
    s3 = get_s3_client()
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(suffix.lower()):
                keys.append(obj["Key"])
    return sorted(keys)


def fetch_object_bytes(bucket: str, key: str) -> bytes:
    """Download any S3 object (CSV or TXT) and return its raw bytes."""
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        raise IOError(
            f"Failed to download s3://{bucket}/{key}: {e}\n"
            f"Please check the bucket name, key, and AWS credentials/permissions."
        )
