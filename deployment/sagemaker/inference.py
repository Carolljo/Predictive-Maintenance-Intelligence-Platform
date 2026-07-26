"""
SageMaker inference entry point for the
Predictive Maintenance Intelligence Platform.

This module loads the trained model artifacts and performs the same
feature engineering and preprocessing used during model training.
"""

import json
import os

import joblib
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the feature engineering used during model training.

    Parameters
    ----------
    df : pd.DataFrame
        Machine observations containing the original Date and sensor
        features expected by the model.

    Returns
    -------
    pd.DataFrame
        Copy of the input data containing the engineered date,
        temperature-difference, and power-index features, with the
        original Date column removed.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter

    df.drop(columns=["Date"], inplace=True)

    df["Temperature_Difference"] = (
        df["Process temperature (K)"]
        - df["Air temperature (K)"]
    )

    df["Power_Index"] = (
        df["Rotational speed (rpm)"]
        * df["Torque (Nm)"]
    )

    return df


def model_fn(model_dir):
    """
    Load the trained artifacts when the SageMaker container starts.

    Parameters
    ----------
    model_dir : str
        Directory where SageMaker extracts the packaged model artifacts.

    Returns
    -------
    dict
        Loaded classification model, preprocessing pipeline,
        and label encoder.

    Side Effects
    ------------
    Reads serialized model artifacts from the SageMaker model directory.
    """

    model = joblib.load(
        os.path.join(model_dir, "best_model.pkl")
    )

    preprocessor = joblib.load(
        os.path.join(model_dir, "preprocessor.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(model_dir, "label_encoder.pkl")
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
    }


def input_fn(request_body, request_content_type):
    """
    Convert an incoming JSON inference request into a DataFrame.

    Parameters
    ----------
    request_body : str
        JSON request body containing one or more machine observations.

    request_content_type : str
        MIME type of the incoming SageMaker request.

    Returns
    -------
    pd.DataFrame
        Machine observations represented as a pandas DataFrame.

    Raises
    ------
    ValueError
        If the request content type is not application/json.
    """

    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    data = json.loads(request_body)

    if isinstance(data, dict):
        data = [data]

    return pd.DataFrame(data)


def predict_fn(input_data, artifacts):
    """
    Generate machine-failure predictions for prepared input data.

    Parameters
    ----------
    input_data : pd.DataFrame
        Machine observations created from the incoming request.

    artifacts : dict
        Loaded model, preprocessing pipeline, and label encoder.

    Returns
    -------
    list
        Prediction dictionaries containing the predicted failure
        category and confidence score for each observation.
    """

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    label_encoder = artifacts["label_encoder"]

    engineered_data = engineer_features(input_data)

    processed_data = preprocessor.transform(
        engineered_data
    )

    encoded_predictions = model.predict(
        processed_data
    )

    predicted_labels = label_encoder.inverse_transform(
        encoded_predictions
    )

    probabilities = model.predict_proba(
        processed_data
    )

    confidence_scores = probabilities.max(axis=1)

    predictions = []

    for label, confidence in zip(
        predicted_labels,
        confidence_scores
    ):
        predictions.append(
            {
                "predicted_failure": str(label),
                "confidence": float(confidence),
            }
        )

    return predictions


def output_fn(prediction, response_content_type):
    """
    Serialize SageMaker predictions as a JSON response.

    Parameters
    ----------
    prediction : list
        Prediction results produced by predict_fn.

    response_content_type : str
        MIME type requested for the SageMaker response.

    Returns
    -------
    str
        JSON-serialized prediction response.

    Raises
    ------
    ValueError
        If the requested response content type is not application/json.
    """

    if response_content_type != "application/json":
        raise ValueError(
            f"Unsupported response content type: "
            f"{response_content_type}"
        )

    return json.dumps(prediction)