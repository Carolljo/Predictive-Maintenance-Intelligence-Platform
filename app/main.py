from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import load_prediction_artifacts, predict_failure


app = FastAPI()


class MachineInput(BaseModel):
    """
    Define the machine sensor data accepted by the local prediction API.

    Attributes:
        Date: Timestamp associated with the machine observation.
        System: Encoded machine system value.
        Control: Machine control category.
        Type: Machine or product type.
        Air_temperature: Ambient air temperature in Kelvin.
        Process_temperature: Machine process temperature in Kelvin.
        Rotational_speed: Machine rotational speed in revolutions per minute.
        Torque: Machine torque in Newton-metres.
        Tool_wear: Accumulated tool wear in minutes.
    """

    Date: str
    System: int
    Control: str
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float


model, preprocessor, label_encoder = load_prediction_artifacts()


@app.get("/")
def home():
    """
    Return a simple health response for the local FastAPI application.

    Returns:
        dict: Message confirming that the Predictive Maintenance API
        is running.
    """

    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(data: MachineInput):
    """
    Predict the diagnostic condition of a machine observation.

    The API input uses simplified field names. They are converted to the
    feature names expected by the trained machine learning pipeline before
    inference is performed.

    Args:
        data: Validated machine sensor and operational data supplied in
        the request body.

    Returns:
        dict: Predicted failure category and the model confidence score.

    Side Effects:
        Uses model artifacts loaded when the FastAPI application starts.
    """

    input_df = pd.DataFrame([{
        "Date": data.Date,
        "System": data.System,
        "Control": data.Control,
        "Type": data.Type,
        "Air temperature (K)": data.Air_temperature,
        "Process temperature (K)": data.Process_temperature,
        "Rotational speed (rpm)": data.Rotational_speed,
        "Torque (Nm)": data.Torque,
        "Tool wear (min)": data.Tool_wear,
    }])

    result = predict_failure(
        input_df,
        model,
        preprocessor,
        label_encoder
    )

    return {
        "predicted_failure": result.iloc[0]["Predicted_Failure"],
        "confidence": float(result.iloc[0]["Confidence"])
    }