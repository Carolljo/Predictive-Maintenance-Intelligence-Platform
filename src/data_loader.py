import pandas as pd


def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded dataset.
    """

    return pd.read_csv(file_path)