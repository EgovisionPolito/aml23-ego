import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.TemporalModel import TemporalModel


class PklDataset(Dataset):
    def __init__(self, file_path):
        try:
            # load data from pickle file and init data attribute/structure
            with open(file_path, "rb") as f:
                self.data = (pickle.load(f))["features"]

        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # ToDo: handle of different features (instead of RGB) is missing
        uid = sample["uid"]
        video_name = sample["video_name"]
        features = torch.tensor(sample["features_RGB"], dtype=torch.float32)
        # features = features.view(32, 5, 1024)  # ? assuming batch_size = 32

        return uid, video_name, features


def aggregate_features():
    extracted_features_path = "saved_features"
    batch_size = 32  # ToDo: adjust batch size, now it's just a random number

    # get list of files in the folder of extracted features (filtering out non .pkl files)
    input_pkl_folder = list(
        filter(lambda file: file.endswith(".pkl"), os.listdir(extracted_features_path))
    )

    for file in input_pkl_folder:
        # * Step 1: Load data from pickle file -> each sample has to be a dictionary with keys: uid, video_name, features(w/ shape 5,1024)
        pkl_dataset = PklDataset(f"{extracted_features_path}/{file}")

        # * Step 2: Create DataLoader
        dataloader = DataLoader(pkl_dataset, batch_size=batch_size, shuffle=True)

        # * Step 3 & 4: Create model and aggregate features along temporal axis
        # ToDo: adjust model parameters (probably the shape of the dataloader should be used (?))
        input_channels = 5
        output_classes = 1
        conv1d_channels = 64
        fc_hidden_units = 128
        sequence_length = 1024 # based on the shape of features
        num_epochs = 1

        model = TemporalModel(
            input_channels,
            output_classes,
            conv1d_channels,
            fc_hidden_units,
            sequence_length,
        )

        aggregated_features = {"features": []}
        for epoch in range(num_epochs):
            temp_features = []
            for batch in dataloader:
                _, _, features = batch

                # Forward pass
                outputs = model(features)

                # ? cpu() -> move data from GPU to CPU, necessary for numpy conversion
                temp_features.append(outputs.detach().cpu().numpy())

        temp_features = np.concatenate(temp_features, axis=0)
        aggregated_features["features"].append(temp_features)

        try:
            with open(f"aggregated_features/aggregated_{file}", "wb") as f:
                pickle.dump(aggregated_features, f)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        print(f"Data from {file} has been successfully aggregated.")
