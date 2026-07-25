import json


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


def lambda_handler(event, context):
    """
    Entry point for the Predictive Maintenance API.
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

        # API Gateway normally sends body as a JSON string
        if isinstance(body, str):
            machine_data = json.loads(body)
        else:
            machine_data = body

        # Body must represent one machine record
        if not isinstance(machine_data, dict):
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Request body must be a JSON object."
                })
            }

        # Check required model input fields
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