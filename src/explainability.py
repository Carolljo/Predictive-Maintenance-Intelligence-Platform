"""
explainability.py

Model explainability utilities for the
Predictive Maintenance Intelligence Platform.
"""

import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
)

from src.model_trainer import load_model
from src.preprocessing import load_preprocessor


def get_feature_names(preprocessor):
    """
    Get transformed feature names from the fitted preprocessor.

    Parameters
    ----------
    preprocessor :
        Fitted preprocessing pipeline.

    Returns
    -------
    list
        List of transformed feature names.
    """

    feature_names = preprocessor.get_feature_names_out()

    return feature_names.tolist()


def get_feature_importance(
    model,
    preprocessor,
) -> pd.DataFrame:
    """
    Extract feature importance values from the trained model.

    Parameters
    ----------
    model :
        Trained Random Forest model.

    preprocessor :
        Fitted preprocessing pipeline.

    Returns
    -------
    pd.DataFrame
        Feature names and their importance scores,
        sorted from highest to lowest importance.
    """

    # Get transformed feature names
    feature_names = get_feature_names(
        preprocessor
    )

    # Get Random Forest feature importance scores
    importance_scores = model.feature_importances_

    # Validate feature alignment
    if len(feature_names) != len(importance_scores):
        raise ValueError(
            "Number of feature names does not match "
            "the number of model feature importances."
        )

    # Create feature importance table
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_scores,
        }
    )

    # Sort from most important to least important
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def get_top_features(
    importance_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the most important model features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance table.

    top_n : int, default=10
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Top model features ranked by importance.
    """

    return importance_df.head(top_n)


if __name__ == "__main__":

    # Load trained artifacts
    model = load_model(
        BEST_MODEL_PATH
    )

    preprocessor = load_preprocessor(
        PREPROCESSOR_PATH
    )

    # Calculate feature importance
    importance_df = get_feature_importance(
        model,
        preprocessor,
    )

    # Display top features
    top_features = get_top_features(
        importance_df,
        top_n=10,
    )

    print("Top 10 Important Features")
    print("-------------------------")
    print(top_features.to_string(index=False))