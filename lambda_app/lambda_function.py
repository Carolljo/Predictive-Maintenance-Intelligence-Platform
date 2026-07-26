"""
AWS Lambda orchestration for the
Predictive Maintenance Intelligence Platform.

The Lambda function validates incoming machine data, invokes the
SageMaker real-time endpoint, sends an SNS notification when a failure
is predicted, and returns the prediction to the API client.
"""

import json

import boto3


# -------------------------------------------------
# Required machine features
# -------------------------------------------------

REQUIRED_FIELDS = [
    "Date",
    "System",
    "Control",
    "Type",
    "Air temperature (K)",
    "Process temperature (K)",
    "Rotational speed (rpm)",
    "Torque (Nm)",
    "Tool wear (min)"
]


# -------------------------------------------------
# AWS configuration
# -------------------------------------------------

AWS_REGION = "ap-south-1"

SNS_TOPIC_ARN = (
    "arn:aws:sns:ap-south-1:686255946915:"
    "predictive-maintenance-alerts"
)

SAGEMAKER_ENDPOINT_NAME = "predictive-maintenance-endpoint"


# -------------------------------------------------
# AWS clients
# -------------------------------------------------

sns_client = boto3.client(
    "sns",
    region_name=AWS_REGION
)

sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=AWS_REGION
)


# -------------------------------------------------
# SageMaker inference
# -------------------------------------------------

def invoke_sagemaker(machine_data):
    """
    Send machine data to the SageMaker real-time endpoint.

    Parameters
    ----------
    machine_data : dict
        Validated machine observation containing the fields expected
        by the SageMaker inference pipeline.

    Returns
    -------
    dict
        First prediction returned by SageMaker, containing the predicted
        failure category and confidence score.

    Raises
    ------
    ValueError
        If SageMaker returns an empty prediction list.

    Side Effects
    ------------
    Makes a network request to the configured SageMaker endpoint.
    """

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(machine_data)
    )

    response_body = (
        response["Body"]
        .read()
        .decode("utf-8")
    )

    predictions = json.loads(response_body)

    if not predictions:
        raise ValueError(
            "SageMaker returned no predictions."
        )

    return predictions[0]


# -------------------------------------------------
# SNS failure alert
# -------------------------------------------------

def send_failure_alert(
    failure_type,
    confidence,
    machine_data
):
    """
    Publish a predicted machine-failure alert to Amazon SNS.

    Parameters
    ----------
    failure_type : str
        Failure category predicted by the machine learning model.

    confidence : float
        Model confidence associated with the prediction.

    machine_data : dict
        Machine observation included in the failure notification.

    Returns
    -------
    dict
        Response returned by the Amazon SNS publish operation.

    Side Effects
    ------------
    Publishes a message to the configured SNS topic. Confirmed
    subscribers may receive an email notification.
    """

    message = (
        "PREDICTIVE MAINTENANCE ALERT\n\n"
        f"Predicted Failure: {failure_type}\n"
        f"Confidence: {confidence:.2%}\n\n"
        "Machine Readings:\n"
        f"Date: {machine_data['Date']}\n"
        f"System: {machine_data['System']}\n"
        f"Control: {machine_data['Control']}\n"
        f"Type: {machine_data['Type']}\n"
        f"Air Temperature: "
        f"{machine_data['Air temperature (K)']} K\n"
        f"Process Temperature: "
        f"{machine_data['Process temperature (K)']} K\n"
        f"Rotational Speed: "
        f"{machine_data['Rotational speed (rpm)']} rpm\n"
        f"Torque: "
        f"{machine_data['Torque (Nm)']} Nm\n"
        f"Tool Wear: "
        f"{machine_data['Tool wear (min)']} min\n"
    )

    response = sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Predictive Maintenance Failure Alert",
        Message=message
    )

    return response


# -------------------------------------------------
# Lambda handler
# -------------------------------------------------

def lambda_handler(event, context):
    """
    Process a prediction request received through AWS Lambda.

    The handler parses and validates the request, invokes the SageMaker
    endpoint, conditionally sends an SNS failure alert, and returns the
    prediction as an HTTP-compatible response.

    Parameters
    ----------
    event : dict
        Lambda event containing the API request body.

    context :
        AWS Lambda runtime context object. It is accepted by the handler
        but is not directly used by the current implementation.

    Returns
    -------
    dict
        HTTP-compatible response containing a status code and JSON body.

        Successful predictions contain the predicted failure category,
        confidence score, and whether an SNS alert was sent.

    Side Effects
    ------------
    Invokes the configured SageMaker endpoint for valid requests.
    Publishes to Amazon SNS when the predicted class is not
    "No failure".
    """

    try:

        # -----------------------------------------
        # Extract request body
        # -----------------------------------------

        body = event.get("body")

        if body is None:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Request body is required."
                })
            }

        # API Gateway normally sends body
        # as a JSON string.
        if isinstance(body, str):
            machine_data = json.loads(body)
        else:
            machine_data = body

        # -----------------------------------------
        # Validate request body
        # -----------------------------------------

        if not isinstance(machine_data, dict):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error":
                    "Request body must be a JSON object."
                })
            }

        # -----------------------------------------
        # Validate required fields
        # -----------------------------------------

        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in machine_data
        ]

        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Missing required fields.",
                    "missing_fields": missing_fields
                })
            }

        # -----------------------------------------
        # Invoke SageMaker
        # -----------------------------------------

        prediction = invoke_sagemaker(
            machine_data
        )

        failure_type = prediction[
            "predicted_failure"
        ]

        confidence = prediction[
            "confidence"
        ]

        # -----------------------------------------
        # Conditional SNS alert
        # -----------------------------------------

        alert_sent = False

        if failure_type != "No failure":

            send_failure_alert(
                failure_type,
                confidence,
                machine_data
            )

            alert_sent = True

        # -----------------------------------------
        # Successful API response
        # -----------------------------------------

        return {
            "statusCode": 200,
            "body": json.dumps({
                "predicted_failure": failure_type,
                "confidence": confidence,
                "alert_sent": alert_sent
            })
        }

    except json.JSONDecodeError:

        return {
            "statusCode": 400,
            "body": json.dumps({
                "error":
                "Invalid JSON in request body."
            })
        }

    except Exception as error:

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(error)
            })
        }