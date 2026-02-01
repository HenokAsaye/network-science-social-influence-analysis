import pandas as pd


def load_network_data(filepath):
    df = pd.read_csv(filepath)
    return df
