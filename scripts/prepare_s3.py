import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/app")

DATASET_DIR = (
    PROJECT_ROOT
    / "customer-support-intent-classification-dataset"
    / "versions"
    / "1"
)

TRAINING_DATA_FILE = DATASET_DIR / "training.csv"
LABEL_MAPPING_FILE = DATASET_DIR / "label_mapping.json"

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME", "public-intent-recognition-system-bucket"
)
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")

S3_TRAINING_PREFIX = os.getenv("S3_TRAINING_PREFIX", "training")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models")
S3_DEPLOYMENT_PREFIX = os.getenv("S3_DEPLOYMENT_PREFIX", "deployment")


# ============================================================
# S3 CLIENT
# ============================================================

def create_s3_client():
    session = boto3.Session(region_name=AWS_REGION)

    if AWS_ENDPOINT_URL:
        print(f"Using custom AWS endpoint: {AWS_ENDPOINT_URL}")
        return session.client("s3", endpoint_url=AWS_ENDPOINT_URL)

    print(f"Using AWS region: {AWS_REGION}")
    return session.client("s3")


# ============================================================
# CHECK BUCKET
# ============================================================

def bucket_exists(s3):
    try:
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        return True
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code in ["404", "NoSuchBucket", "NotFound"]:
            return False
        raise


# ============================================================
# CREATE BUCKET
# ============================================================

def create_bucket(s3):
    if bucket_exists(s3):
        print(f"Bucket already exists: {S3_BUCKET_NAME}")
        return

    print(f"Creating bucket: {S3_BUCKET_NAME}")

    # us-east-1 is the only region that rejects LocationConstraint.
    # All other regions (including eu-west-1 on LocalStack) require it.
    if AWS_REGION == "us-east-1":
        s3.create_bucket(Bucket=S3_BUCKET_NAME)
    else:
        s3.create_bucket(
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={
                "LocationConstraint": AWS_REGION
            },
        )

    print("Bucket created successfully.")


# ============================================================
# CREATE PREFIX
# ============================================================

def create_prefix(s3, prefix):
    key = prefix.rstrip("/") + "/"
    print(f"Creating S3 prefix: {key}")
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=b"")


# ============================================================
# UPLOAD FILE
# ============================================================

def upload_file(s3, local_file, s3_key):
    local_file = Path(local_file)

    if not local_file.exists():
        raise FileNotFoundError(f"File does not exist: {local_file}")

    print(f"Uploading: {local_file}")
    print(f"         -> s3://{S3_BUCKET_NAME}/{s3_key}")

    s3.upload_file(str(local_file), S3_BUCKET_NAME, s3_key)
    print("Upload completed.")


# ============================================================
# VERIFY OBJECT
# ============================================================

def verify_object(s3, s3_key):
    try:
        response = s3.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        size = response.get("ContentLength", 0)
        print(f"Verified: s3://{S3_BUCKET_NAME}/{s3_key}")
        print(f"Size: {size} bytes")
        return True
    except ClientError:
        print(f"WARNING: Object could not be verified: {s3_key}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("S3 DATA PREPARATION")
    print("=" * 60)
    print()

    print(f"Bucket: {S3_BUCKET_NAME}")
    print(f"Region: {AWS_REGION}")
    if AWS_ENDPOINT_URL:
        print(f"Endpoint: {AWS_ENDPOINT_URL}")
    print()

    print("Checking local training files...")
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAINING_DATA_FILE}"
        )
    if not LABEL_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Label mapping not found:\n{LABEL_MAPPING_FILE}"
        )

    print("Training files found.\n")

    s3 = create_s3_client()
    create_bucket(s3)
    print()

    create_prefix(s3, S3_TRAINING_PREFIX)
    create_prefix(s3, S3_MODEL_PREFIX)
    create_prefix(s3, S3_DEPLOYMENT_PREFIX)
    print()

    training_key = f"{S3_TRAINING_PREFIX}/training.csv"
    upload_file(s3, TRAINING_DATA_FILE, training_key)

    label_mapping_key = f"{S3_TRAINING_PREFIX}/label_mapping.json"
    upload_file(s3, LABEL_MAPPING_FILE, label_mapping_key)
    print()

    print("Verifying uploaded files...")
    training_verified = verify_object(s3, training_key)
    label_mapping_verified = verify_object(s3, label_mapping_key)
    print()

    if training_verified and label_mapping_verified:
        print("=" * 60)
        print("S3 PREPARATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print()
        print(f"Training data:\ns3://{S3_BUCKET_NAME}/{training_key}\n")
        print(f"Label mapping:\ns3://{S3_BUCKET_NAME}/{label_mapping_key}\n")
    else:
        print("=" * 60)
        print("S3 PREPARATION FAILED")
        print("=" * 60)
        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()