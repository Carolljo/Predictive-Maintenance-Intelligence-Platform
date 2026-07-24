from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import load_prediction_artifacts, predict_failure


app = FastAPI()


class MachineInput(BaseModel):
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
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(data: MachineInput):

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