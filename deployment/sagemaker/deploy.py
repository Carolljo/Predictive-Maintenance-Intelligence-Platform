"""
deploy.py

Deploy the Predictive Maintenance model to
an Amazon SageMaker real-time endpoint.
"""

import boto3
import sagemaker

from sagemaker.sklearn.model import SKLearnModel


# -------------------------------------------------
# AWS configuration
# -------------------------------------------------

AWS_PROFILE = "predictive-maintenance"
AWS_REGION = "ap-south-1"

ROLE_ARN = (
    "arn:aws:iam::686255946915:"
    "role/PredictiveMaintenanceSageMakerRole"
)

MODEL_DATA = (
    "s3://carol-predictive-maintenance/"
    "model/model.tar.gz"
)

ENDPOINT_NAME = "predictive-maintenance-endpoint"


# -------------------------------------------------
# AWS / SageMaker sessions
# -------------------------------------------------

boto_session = boto3.Session(
    profile_name=AWS_PROFILE,
    region_name=AWS_REGION,
)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session
)


# -------------------------------------------------
# Define SageMaker model
# -------------------------------------------------

model = SKLearnModel(
    model_data=MODEL_DATA,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="deployment/sagemaker",
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=sagemaker_session,
)


# -------------------------------------------------
# Deploy real-time endpoint
# -------------------------------------------------

if __name__ == "__main__":

    print("Starting SageMaker deployment...")
    print(f"Region: {AWS_REGION}")
    print(f"Model artifact: {MODEL_DATA}")
    print(f"Endpoint: {ENDPOINT_NAME}")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME,
    )

    print("\nDeployment completed successfully.")
    print(f"Endpoint: {predictor.endpoint_name}")