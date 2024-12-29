import csv
import os
import pandas as pd
from django.conf import settings


def load_csv_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file not found
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of other errors
