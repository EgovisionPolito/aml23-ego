import torch
import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

extracted_features_path = "saved_features"


class PklDataset(Dataset):
    def __init__(self, pickle_file):
        try:
            # load data from pickle file and init data attribute/structure
            with open(f"{extracted_features_path}/{pickle_file}", "rb") as f:
                self.data = pickle.load(f)

        except FileNotFoundError:
            print(f"Error: File {pickle_file} not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data["features"][index]

        # ToDo: handle of different features (instead of RGB) is missing
        uid = sample["uid"]
        video_name = sample["video_name"]
        features = torch.tensor(sample["features_RGB"], dtype=torch.float32)

        return uid, video_name, features


def aggregate_features():
    batch_size = 32  # ToDo: adjust batch size, now it's just a random number

    # get list of files in the folder of extracted features (filtering out non .pkl files)
    input_pkl_folder = list(
        filter(lambda file: file.endswith(".pkl"), os.listdir(extracted_features_path))
    )

    try:
        for file in input_pkl_folder:
            # * Step 1: Load data from pickle file
            pkl_dataset = PklDataset(file)

            """ # Example of how to use the dataset
            for uid, video_name, features in pkl_dataset:
                # do something
            """

            # * Step 2: Create DataLoader
            data_loader = DataLoader(pkl_dataset, batch_size=batch_size, shuffle=True)

            print(f"Data from {file} has been successfully aggregated.")
    except FileNotFoundError:
        print(f"Error: File {input_pkl_folder} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


#! TEST purpose only -> should be removed
if __name__ == "__main__":
    aggregate_features()
