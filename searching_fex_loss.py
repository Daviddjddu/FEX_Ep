import numpy as np

# Define the file path
file_path_S = "Dim0/Searching_S.txt"
file_path_I = "Dim1/Searching_I.txt"
file_path_R = "Dim2/Searching_R.txt"

# Read the file and extract the fifth column
data_S = []
data_I = []
data_R = []

with open(file_path_S, "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        items = line.strip().split()  # Split by whitespace
        if len(items) >= 5:  # Ensure there are enough columns
            data_S.append(float(items[4]))  # Extract the fifth column (index 4)

with open(file_path_I, "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        items = line.strip().split()  # Split by whitespace
        if len(items) >= 5:  # Ensure there are enough columns
            data_I.append(float(items[4]))  # Extract the fifth column (index 4)
            
with open(file_path_R, "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        items = line.strip().split()  # Split by whitespace
        if len(items) >= 5:  # Ensure there are enough columns
            data_R.append(float(items[4]))  # Extract the fifth column (index 4)
            
# Convert list to numpy array
Searching_S = np.array(data_S)
Searching_I = np.array(data_I)
Searching_R = np.array(data_R)

Searching_mean = np.mean([Searching_S, Searching_I, Searching_R], axis=0)

np.save("Data/sir_searching_fex.npy", Searching_mean)

