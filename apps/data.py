import pandas as pd

# File paths
file_path_1 = "/Users/blueaivee/Desktop/Streamlit_hybrid_devop/Datasets/ireland_processed.csv"
file_path_2 = "/Users/blueaivee/Desktop/Streamlit_hybrid_devop/Datasets/uk_processed.csv"


def load_ireland_data():
    return pd.read_csv(file_path_1, parse_dates=['Date'])

def load_uk_data():
    return pd.read_csv(file_path_2, parse_dates=['Date'])