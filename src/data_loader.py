import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    ENGINEERED_DATA_PATH,
)

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)