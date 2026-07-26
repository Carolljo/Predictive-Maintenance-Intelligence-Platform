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
    Send machine data to the SageMaker endpoint
    and return the model prediction.
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
    Publish a machine failure alert to the SNS topic.
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
    Entry point for the Predictive Maintenance API.

    Flow:
    1. Receive API request.
    2. Parse JSON body.
    3. Validate required machine fields.
    4. Invoke SageMaker endpoint.
    5. Read model prediction.
    6. Send SNS alert when failure is predicted.
    7. Return prediction to the client.
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