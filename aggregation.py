import numpy as np
from utils.temporal_aggregation import aggregate_features
from utils.extract_pkl import *

pkl_folder = [
    "aggregated_features/aggregated_test_D5",
    "aggregated_features/aggregated_test_D10",
    "aggregated_features/aggregated_test_D25",
    "aggregated_features/aggregated_test_U5",
    "aggregated_features/aggregated_test_U10",
    "aggregated_features/aggregated_test_U25",
]

if __name__ == "__main__":
    aggregate_features()
    extract_pkl(pkl_folder)

    data = np.array((get_data_from_pkl("saved_features/test_D5"))['features'])
    data_aggregated = np.array((get_data_from_pkl(pkl_folder[0]))['features'])

    print(f"\nData shape: {data.shape}")
    print(f"Data aggregated shape: {data_aggregated.shape}")
