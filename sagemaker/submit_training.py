import os
import boto3

import sagemaker
from sagemaker.estimator import Estimator


# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "eu-west-1"
)

S3_BUCKET = (
    "public-intent-recognition-system-bucket"
)

S3_TRAINING_PREFIX = (
    "training"
)

S3_OUTPUT_PREFIX = (
    "models"
)

TRAINING_IMAGE_URI = os.getenv(
    "TRAINING_IMAGE_URI"
)

SAGEMAKER_ROLE_ARN = os.getenv(
    "SAGEMAKER_ROLE_ARN"
)


# ============================================================
# S3 LOCATIONS
# ============================================================

TRAINING_S3_URI = (
    f"s3://{S3_BUCKET}/"
    f"{S3_TRAINING_PREFIX}/"
)

OUTPUT_S3_URI = (
    f"s3://{S3_BUCKET}/"
    f"{S3_OUTPUT_PREFIX}/"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_configuration():

    if not TRAINING_IMAGE_URI:

        raise ValueError(
            "TRAINING_IMAGE_URI environment "
            "variable is not configured."
        )

    if not SAGEMAKER_ROLE_ARN:

        raise ValueError(
            "SAGEMAKER_ROLE_ARN environment "
            "variable is not configured."
        )


# ============================================================
# SUBMIT TRAINING JOB
# ============================================================

def submit_training_job():

    validate_configuration()

    print("=" * 60)
    print("SAGEMAKER TRAINING JOB SUBMISSION")
    print("=" * 60)

    print(
        f"AWS Region: {AWS_REGION}"
    )

    print(
        f"Training image: "
        f"{TRAINING_IMAGE_URI}"
    )

    print(
        f"Training data: "
        f"{TRAINING_S3_URI}"
    )

    print(
        f"Model output: "
        f"{OUTPUT_S3_URI}"
    )

    # --------------------------------------------------------
    # Boto3 session
    # --------------------------------------------------------

    boto_session = boto3.Session(
        region_name=AWS_REGION
    )

    # --------------------------------------------------------
    # SageMaker session
    # --------------------------------------------------------

    sagemaker_session = (
        sagemaker.Session(
            boto_session=boto_session
        )
    )

    # --------------------------------------------------------
    # Create estimator
    # --------------------------------------------------------

    estimator = Estimator(

        image_uri=TRAINING_IMAGE_URI,

        role=SAGEMAKER_ROLE_ARN,

        instance_count=1,

        instance_type="ml.m5.large",

        output_path=OUTPUT_S3_URI,

        sagemaker_session=sagemaker_session,

        base_job_name=(
            "intent-recognition-training"
        ),

        hyperparameters={

            "model-name":
                "distilbert-base-uncased",

            "training-command":
                "train",

            "dataset-file":
                "training.csv",

            "label-mapping-file":
                "label_mapping.json",

            "test-size":
                0.2,

            "random-state":
                42,

            "num-train-epochs":
                3,

            "per-device-train-batch-size":
                8,

            "per-device-eval-batch-size":
                8,

            "weight-decay":
                0.01,

            "learning-rate":
                2e-5,

            "eval-strategy":
                "epoch",

            "save-strategy":
                "epoch",

            "logging-strategy":
                "steps",

            "logging-steps":
                50,

            "load-best-model-at-end":
                True,

            "metric-for-best-model":
                "accuracy",

            "greater-is-better":
                True
        }
    )

    # --------------------------------------------------------
    # Input channel
    # --------------------------------------------------------

    train_input = (
        sagemaker.inputs.TrainingInput(
            s3_data=TRAINING_S3_URI,
            content_type="text/csv",
            input_mode="File"
        )
    )

    # --------------------------------------------------------
    # Submit job
    # --------------------------------------------------------

    print()
    print(
        "Submitting SageMaker training job..."
    )

    estimator.fit(

        inputs={
            "train": train_input
        },

        wait=True,

        logs=True
    )

    print()
    print("=" * 60)
    print("TRAINING JOB COMPLETED")
    print("=" * 60)

    print(
        f"Training job: "
        f"{estimator.latest_training_job.name}"
    )

    print(
        f"Model artifact: "
        f"{estimator.model_data}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    submit_training_job()