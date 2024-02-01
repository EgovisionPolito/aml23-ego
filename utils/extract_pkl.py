import pickle

# * Setup for extrated features

input_pkl_folder = [
    "saved_features/test_D5.pkl",
    "saved_features/test_D10.pkl",
    "saved_features/test_D25.pkl",
    "saved_features/test_U5.pkl",
    "saved_features/test_U10.pkl",
    "saved_features/test_U25.pkl",
]

output_txt_folder = [
    "saved_features/test_D5.txt",
    "saved_features/test_D10.txt",
    "saved_features/test_D25.txt",
    "saved_features/test_U5.txt",
    "saved_features/test_U10.txt",
    "saved_features/test_U25.txt",
]

try:
    for i in range(len(input_pkl_folder)):
        # Open the .pkl file in binary mode for reading
        with open(input_pkl_folder[i], "rb") as pkl_file:
            # Load the data from the .pkl file
            data = pickle.load(pkl_file)

        # Open the .txt file in text mode for writing
        with open(output_txt_folder[i], "w") as txt_file:
            # Write the extracted data to the .txt file
            txt_file.write(str(data))

        print(
            f"Data from {input_pkl_folder[i]} has been successfully extracted and saved to {output_txt_folder[i]}."
        )
except FileNotFoundError:
    print(f"Error: File {input_pkl_folder} not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
