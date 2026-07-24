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
    Build the preprocessing pipeline for numerical and categorical features.

    Parameters
    ----------
    numerical_features : List[str]
        Names of the numerical features.

    categorical_features : List[str]
        Names of the categorical features.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing pipeline containing numerical and
        categorical transformations.
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
):
    """
    Transform feature data using a fitted preprocessing pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.

    X : pd.DataFrame
        Feature data to transform.

    Returns
    -------
    array-like
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
        Preprocessing pipeline to fit.

    X_train : pd.DataFrame
        Training feature data used to fit and apply the transformations.

    Returns
    -------
    array-like
        Transformed training feature matrix.
    """

    return preprocessor.fit_transform(X_train)


def save_preprocessor(
    preprocessor: ColumnTransformer,
    file_path: str
) -> None:
    """
    Save a fitted preprocessing pipeline to disk.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline to save.

    file_path : str
        Path where the preprocessing pipeline will be saved.

    Returns
    -------
    None
        The function saves the preprocessor to disk and does not
        return a value.
    """

    save_object(preprocessor, file_path)


def load_preprocessor(
    file_path: str
) -> ColumnTransformer:
    """
    Load a fitted preprocessing pipeline from disk.

    Parameters
    ----------
    file_path : str
        Path to the saved preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        Loaded fitted preprocessing pipeline.
    """

    return load_object(file_path)