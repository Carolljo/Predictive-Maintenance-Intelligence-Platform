"""
model_evaluator.py

Reusable model evaluation utilities for the Predictive Maintenance project.
"""

from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(
    model,
    X_test,
    y_test
) -> Dict[str, Any]:
    """
    Evaluate a trained classification model.

    Parameters
    ----------
    model :
        Trained classification model.

    X_test :
        Test feature data.

    y_test :
        True target values.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predictions and evaluation metrics.
    """

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # Generate confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # Store all evaluation results
    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": y_pred,
    }

    return results