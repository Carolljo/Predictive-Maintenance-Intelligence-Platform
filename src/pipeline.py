"""
pipeline.py

End-to-end training pipeline for the
Predictive Maintenance Intelligence Platform.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLEANED_DATA_PATH,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    LABEL_ENCODER_PATH,
)

from src.data_loader import load_data
from src.feature_engineering import engineer_features

from src.preprocessing import (
    build_preprocessor,
    fit_transform_data,
    transform_data,
    save_preprocessor,
)

from src.model_trainer import (
    train_model,
    save_model,
)

from src.model_evaluator import evaluate_model

from src.utils import save_object


def run_training_pipeline():
    """
    Run the complete model training pipeline.

    Steps:
        1. Load cleaned data
        2. Engineer features
        3. Separate features and target
        4. Encode target labels
        5. Split training and test data
        6. Build and fit preprocessing pipeline
        7. Transform training and test data
        8. Train Random Forest model
        9. Evaluate model
        10. Save model and preprocessing artifacts
    """

    print("Starting training pipeline...")

    # -------------------------------------------------
    # 1. Load cleaned dataset
    # -------------------------------------------------

    df = load_data(CLEANED_DATA_PATH)

    print(f"Loaded dataset: {df.shape}")

    # -------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------

    df = engineer_features(df)

    print(f"Dataset after feature engineering: {df.shape}")

    # -------------------------------------------------
    # 3. Separate features and target
    # -------------------------------------------------

    X = df.drop(columns=["Diagnostic"])
    y = df["Diagnostic"]

    # -------------------------------------------------
    # 4. Encode target labels
    # -------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # -------------------------------------------------
    # 5. Train-test split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # -------------------------------------------------
    # 6. Identify numerical and categorical features
    # -------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # -------------------------------------------------
    # 7. Build preprocessing pipeline
    # -------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Fit only on training data
    X_train_processed = fit_transform_data(
        preprocessor,
        X_train,
    )

    # Transform test data using fitted preprocessor
    X_test_processed = transform_data(
        preprocessor,
        X_test,
    )

    # -------------------------------------------------
    # 8. Train model
    # -------------------------------------------------

    model = train_model(
        X_train_processed,
        y_train,
    )

    print("Model training completed.")

    # -------------------------------------------------
    # 9. Evaluate model
    # -------------------------------------------------

    results = evaluate_model(
        model,
        X_test_processed,
        y_test,
    )

    print("\nModel Evaluation")
    print("----------------------------")

    print(
        f"Accuracy: {results['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {results['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: {results['weighted_f1']:.4f}"
    )

    print("\nClassification Report:")
    print(results["classification_report"])

    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    # -------------------------------------------------
    # 10. Save artifacts
    # -------------------------------------------------

    save_model(
        model,
        BEST_MODEL_PATH,
    )

    save_preprocessor(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    save_object(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    print("\nArtifacts saved successfully.")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Label encoder: {LABEL_ENCODER_PATH}")

    print("\nTraining pipeline completed successfully.")

    return results


if __name__ == "__main__":
    run_training_pipeline()