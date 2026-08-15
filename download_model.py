from pathlib import Path
from dotenv import load_dotenv
import os
import shutil

from manage_s3_buckets import S3_bucket_crud

load_dotenv()

accuracy_comparison_file = os.getenv("accuracy_comparison_file")

current_model_dir = Path(__file__).resolve().parent / "current_model"

if current_model_dir.exists():
    shutil.rmtree(current_model_dir)

current_model_dir.mkdir(parents=True)

def download_latest_model():

    s3 = S3_bucket_crud(
        bucket_name=os.getenv("s3_bucket_name"),
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region_name"),
    )

    current_model_dir = Path(__file__).resolve().parent / "current_model"

    current_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        (
            s3_model_path,
            directory_name,
            accuracy,
        ) = s3.read_file_from_s3(accuracy_comparison_file)

    except ClientError as e:

        if e.response["Error"]["Code"] == "NoSuchKey":

            print(
                "No existing model pointer found in S3."
            )

            print(
                "This is the first training run. "
                "Training from scratch."
            )

            return None

        raise

    if not s3_model_path:

        print(
            "No model path found in S3 pointer."
        )

        print(
            "Training from scratch."
        )

        return None

    print(f"Downloading model: {s3_model_path}")

    tag = s3.download_folder(
        s3_model_path,
        current_model_dir,
    )

    if tag:

        print(
            f"Model downloaded to: {current_model_dir}"
        )

        return current_model_dir

    print("Failed to download existing model.")

    return None
if __name__ == "__main__":
    download_latest_model()