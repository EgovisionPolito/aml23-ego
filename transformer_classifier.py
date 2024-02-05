import numpy as np
from models.FinalClassifier import TransformerClassifier
from utils.extract_pkl import *

pkl_folder = [
    "saved_features/aggregated_test_D5",
    "saved_features/aggregated_test_D10",
    "saved_features/aggregated_test_D25",
    "saved_features/aggregated_test_U5",
    "saved_features/aggregated_test_U10",
    "saved_features/aggregated_test_U25",
]

if __name__ == "__main__":
    print("test")
