"""
predict.py

Prediction utilities for the Predictive Maintenance
Intelligence Platform.
"""

import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    LABEL_ENCODER_PATH,
)

from src.feature_engineering import engineer_features
from src.preprocessing import (
    load_preprocessor,
    transform_data,
)
from src.model_trainer import load_model
from src.utils import load_object


def load_prediction_artifacts():
    """
    Load all artifacts required for prediction.

    Returns:
        tuple: Trained model, fitted preprocessor,
               and fitted label encoder.
    """

    model = load_model(BEST_MODEL_PATH)

    preprocessor = load_preprocessor(
        PREPROCESSOR_PATH
    )

    label_encoder = load_object(
        LABEL_ENCODER_PATH
    )

    return model, preprocessor, label_encoder


def predict_failure(
    input_data: pd.DataFrame,
    model,
    preprocessor,
    label_encoder,
):
    """
    Predict the machine failure type for new input data.

    Parameters
    ----------
    input_data : pd.DataFrame
        New machine data in the same format as the
        cleaned training data, excluding Diagnostic.

    model :
        Trained classification model.

    preprocessor :
        Fitted preprocessing pipeline.

    label_encoder :
        Fitted target label encoder.

    Returns
    -------
    pd.DataFrame
        Prediction results containing predicted
        class and confidence score.
    """

    # Work on a copy to avoid modifying original input
    data = input_data.copy()

    # Apply the same feature engineering used during training
    data = engineer_features(data)

    # Apply the fitted preprocessing transformations
    processed_data = transform_data(
        preprocessor,
        data
    )

    # Generate encoded predictions
    encoded_predictions = model.predict(
        processed_data
    )

    # Convert encoded predictions back to failure names
    predicted_labels = label_encoder.inverse_transform(
        encoded_predictions
    )

    # Get prediction probabilities
    probabilities = model.predict_proba(
        processed_data
    )

    # Highest probability for each prediction
    confidence_scores = probabilities.max(
        axis=1
    )

    # Create readable prediction output
    results = pd.DataFrame(
        {
            "Predicted_Failure": predicted_labels,
            "Confidence": confidence_scores,
        }
    )

    return results


if __name__ == "__main__":

    model, preprocessor, label_encoder = (
        load_prediction_artifacts()
    )

    print("Prediction artifacts loaded successfully.")