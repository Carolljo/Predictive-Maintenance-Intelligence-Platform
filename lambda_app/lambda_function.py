import json
import boto3


# Required machine features expected by the ML pipeline
REQUIRED_FIELDS = [
    "System",
    "Control",
    "Type",
    "Air temperature (K)",
    "Process temperature (K)",
    "Rotational speed (rpm)",
    "Torque (Nm)",
    "Tool wear (min)"
]


# SNS configuration
SNS_TOPIC_ARN = (
    "arn:aws:sns:ap-south-1:686255946915:"
    "predictive-maintenance-alerts"
)

sns_client = boto3.client(
    "sns",
    region_name="ap-south-1"
)


def send_failure_alert(failure_type, confidence, machine_data):
    """
    Publish a machine failure alert to the SNS topic.
    """

    message = (
        "PREDICTIVE MAINTENANCE ALERT\n\n"
        f"Predicted Failure: {failure_type}\n"
        f"Confidence: {confidence:.2%}\n\n"
        "Machine Readings:\n"
        f"System: {machine_data['System']}\n"
        f"Control: {machine_data['Control']}\n"
        f"Type: {machine_data['Type']}\n"
        f"Air Temperature: "
        f"{machine_data['Air temperature (K)']} K\n"
        f"Process Temperature: "
        f"{machine_data['Process temperature (K)']} K\n"
        f"Rotational Speed: "
        f"{machine_data['Rotational speed (rpm)']} rpm\n"
        f"Torque: {machine_data['Torque (Nm)']} Nm\n"
        f"Tool Wear: {machine_data['Tool wear (min)']} min\n"
    )

    response = sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="Predictive Maintenance Failure Alert",
        Message=message
    )

    return response


def lambda_handler(event, context):
    """
    Entry point for the Predictive Maintenance API.

    Current responsibilities:
    1. Receive the request.
    2. Parse the request body.
    3. Validate required machine fields.

    The next integration step will:
    4. Invoke the SageMaker endpoint.
    5. Read the model prediction.
    6. Send an SNS alert when a failure is predicted.
    """

    try:
        # Extract request body
        body = event.get("body")

        if body is None:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Request body is required."
                })
            }

        # API Gateway normally sends the body as a JSON string
        if isinstance(body, str):
            machine_data = json.loads(body)
        else:
            machine_data = body

        # Body must contain one machine record
        if not isinstance(machine_data, dict):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Request body must be a JSON object."
                })
            }

        # Check required fields
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

        # SageMaker inference will be added here next.
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Machine data validated successfully.",
                "data": machine_data
            })
        }

    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Invalid JSON in request body."
            })
        }

    except Exception as error:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(error)
            })
        }