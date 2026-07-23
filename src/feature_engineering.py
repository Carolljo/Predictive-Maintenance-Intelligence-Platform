import pandas as pd


def engineer_features(df):
    """
    Perform feature engineering on the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with engineered features.
    """

    df = df.copy()
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Extract date-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    # Extract the day of the week
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    
    # Extract the quarter of the year
    df["Quarter"] = df["Date"].dt.quarter
    
    # Remove the original Date column
    df.drop(columns=["Date"], inplace=True)
    
    # Calculate the temperature difference
    df["Temperature_Difference"] = (
    df["Process temperature (K)"] - df["Air temperature (K)"])
    
    # Calculate the power index
    df["Power_Index"] = (
    df["Rotational speed (rpm)"] * df["Torque (Nm)"])
    
    
    
    return df