"""
preprocessing.py

Reusable preprocessing utilities for the Predictive Maintenance project.
"""

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object, load_object


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build and return the preprocessing pipeline.

    Parameters
    ----------
    numerical_features : List[str]
        List of numerical feature names.

    categorical_features : List[str]
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline.
    """

    # Numerical preprocessing pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame
) -> ColumnTransformer:
    """
    Fit the preprocessing pipeline on the training data.
    """

    preprocessor.fit(X_train)

    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
):
    """
    Transform data using a fitted preprocessing pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.

    X : pd.DataFrame
        Feature data to transform.

    Returns
    -------
    Transformed feature matrix.
    """

    return preprocessor.transform(X)


def fit_transform_data(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame
):
    """
    Fit the preprocessing pipeline and transform the training data.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing pipeline.

    X_train : pd.DataFrame
        Training feature data.

    Returns
    -------
    Transformed training feature matrix.
    """

    return preprocessor.fit_transform(X_train)


def save_preprocessor(
    preprocessor: ColumnTransformer,
    file_path: str
) -> None:
    """
    Save the fitted preprocessing pipeline.
    """

    save_object(preprocessor, file_path)


def load_preprocessor(
    file_path: str
) -> ColumnTransformer:
    """
    Load a saved preprocessing pipeline.
    """

    return load_object(file_path)