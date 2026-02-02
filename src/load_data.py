import pandas as pd
"""
    Loads network data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the network data.
"""

def load_network_data(filepath):
    df = pd.read_csv(filepath)
    return df
