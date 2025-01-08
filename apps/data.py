import pandas as pd
import os

# File paths
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path_1 = os.path.join(base_dir, '../Datasets/ireland_processed.csv')
file_path_2 = os.path.join(base_dir, '../Datasets/uk_processed.csv')

def load_ireland_data():
    return pd.read_csv(file_path_1, parse_dates=['Date'])

def load_uk_data():
    return pd.read_csv(file_path_2, parse_dates=['Date'])
