"""
Project configuration file.

This module stores all directory and file paths used throughout
the Predictive Maintenance Intelligence Platform.
"""


# Directory Paths


ARTIFACTS_DIR = "artifacts"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"



# Data Paths


RAW_DATA_PATH = "data/raw/AI4I-PMDI.csv"

CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"

ENGINEERED_DATA_PATH = "data/processed/engineered_data.csv"



# Model Artifact Paths


# Baseline model selected in Notebook 06
BEST_MODEL_PATH = "artifacts/best_model.pkl"

# Tuned model from Notebook 07
TUNED_MODEL_PATH = "artifacts/tuned_random_forest.pkl"

PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

LABEL_ENCODER_PATH = "artifacts/label_encoder.pkl"