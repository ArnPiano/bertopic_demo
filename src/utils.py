import pandas as pd

def report_missing_values(df: pd.DataFrame):
    """
    Return the number of missing values in each column.
    Only columns with missing values are included.
    """
    missing = df.isnull().sum()
    return missing

def check_duplicates(series: pd.Series) -> bool:
    """
    Check whether a pandas Series contains duplicate values.

    Args:
        series: The pandas Series to check.

    Returns:
        True if duplicates exist, False otherwise.
    """
    return series.duplicated().any()
