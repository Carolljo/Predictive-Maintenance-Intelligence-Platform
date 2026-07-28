"""
Streamlit frontend for the Predictive Maintenance Intelligence Platform.

The application collects machine operating data from the user and sends
the readings to the deployed AWS API Gateway endpoint. The API invokes
the backend prediction pipeline and returns the predicted failure type,
prediction confidence, and SNS alert status.
"""

import requests
import streamlit as st


# -------------------------------------------------
# Application configuration
# -------------------------------------------------

API_URL = (
    "https://xzlyssw5y8.execute-api.ap-south-1.amazonaws.com/predict"
)

st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
)


# -------------------------------------------------
# API communication
# -------------------------------------------------

def request_prediction(machine_data):
    """
    Send machine operating data to the deployed prediction API.

    Args:
        machine_data (dict):
            Machine information and sensor readings required by
            the predictive maintenance model.

    Returns:
        requests.Response:
            HTTP response returned by the API Gateway endpoint.

    Raises:
        requests.exceptions.RequestException:
            If the API cannot be reached or the request fails.
    """

    return requests.post(
        API_URL,
        json=machine_data,
        timeout=30
    )


# -------------------------------------------------
# Page header
# -------------------------------------------------

st.title("Predictive Maintenance Intelligence Platform")

st.write(
    "Monitor machine operating conditions and predict potential "
    "equipment failures."
)

st.subheader("Machine Condition Analysis")


# -------------------------------------------------
# Machine information
# -------------------------------------------------

st.markdown("### Machine Information")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date")

    system = st.number_input(
        "System",
        min_value=0,
        step=1
    )

with col2:
    control = st.selectbox(
        "Control",
        ["A", "B", "C"]
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )


# -------------------------------------------------
# Sensor readings
# -------------------------------------------------

st.markdown("### Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    air_temperature = st.number_input(
        "Air Temperature (K)",
        value=298.0
    )

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        value=1500.0
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=0.0
    )

with col2:
    process_temperature = st.number_input(
        "Process Temperature (K)",
        value=308.0
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

analyze_button = st.button(
    "Analyze Machine",
    type="primary",
    use_container_width=True
)

if analyze_button:

    # Build the JSON payload expected by the Lambda backend.
    machine_data = {
        "Date": str(date),
        "System": int(system),
        "Control": control,
        "Type": machine_type,
        "Air temperature (K)": float(air_temperature),
        "Process temperature (K)": float(process_temperature),
        "Rotational speed (rpm)": float(rotational_speed),
        "Torque (Nm)": float(torque),
        "Tool wear (min)": float(tool_wear),
    }

    try:
        # Send machine readings through API Gateway for prediction.
        with st.spinner("Analyzing machine condition..."):
            response = request_prediction(machine_data)

        if response.status_code == 200:
            result = response.json()

            failure_type = result["predicted_failure"]
            confidence = result["confidence"]
            alert_sent = result["alert_sent"]

            # Display prediction returned by the AWS backend.
            st.markdown("### Prediction Result")

            if failure_type == "No failure":
                st.success("Machine operating normally")
            else:
                st.error(
                    f"Machine Failure Detected: {failure_type}"
                )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Prediction",
                failure_type
            )

            col2.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

            col3.metric(
                "Alert Sent",
                "Yes" if alert_sent else "No"
            )

        else:
            st.error(
                f"API request failed with status "
                f"{response.status_code}: {response.text}"
            )

    except requests.exceptions.Timeout:
        st.error(
            "The prediction service timed out. Please try again."
        )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Unable to connect to the prediction service: {error}"
        )

    except (KeyError, ValueError) as error:
        st.error(
            f"Invalid response from prediction service: {error}"
        )