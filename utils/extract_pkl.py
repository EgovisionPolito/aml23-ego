import pickle


def extract_pkl(pkl_folder):
    """Example of pkl_folder:
    pkl_folder = [
        "saved_features/test_D5",
        "saved_features/test_D10",
        "saved_features/test_D25",
        "saved_features/test_U5",
        "saved_features/test_U10",
        "saved_features/test_U25",
    ]
    """

    try:
        for i in range(len(pkl_folder)):
            # Open the .pkl file in binary mode for reading
            with open(f"{pkl_folder[i]}.pkl", "rb") as pkl_file:
                # Load the data from the .pkl file
                data = pickle.load(pkl_file)

            # Open the .txt file in text mode for writing
            with open(f"{pkl_folder[i]}.txt", "w") as txt_file:
                # Write the extracted data to the .txt file
                txt_file.write(str(data))

            print(
                f"Data from {pkl_folder[i]}.pkl has been successfully extracted and saved to {pkl_folder[i]}.txt."
            )
    except FileNotFoundError:
        print(f"Error: File {pkl_folder} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
