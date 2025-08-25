import numpy as np
import matplotlib.pyplot as plt

# Load the saved .npy file
Searching_loss_fex = np.load("Data/sir_searching_fex.npy")
Training_loss_NN = np.load("Data/sir_train_losses_NN.npy")
Training_loss_RNN = np.load("Data/sir_train_losses_RNN.npy")

# Define x-values as integers from 1 to 100
x_values = list(range(1, 101))  # Ensuring it's a list of integers
x_values_1 = list(range(1, len(Training_loss_RNN)+1))

# # Plot the data
plt.figure(figsize=(12, 8))
plt.plot(x_values, Searching_loss_fex, label="FEX", color = "blue", linewidth = 5)
plt.plot(x_values_1, Training_loss_RNN, label="RNN", color = "Orange", linewidth = 5)
plt.plot(x_values, Training_loss_NN, label="NN", color = "Green", linewidth = 5)
plt.xlabel("Epoch", fontsize = 28)
plt.ylabel("MSE", fontsize = 28)
plt.yscale("log")
plt.xticks(fontsize=28)  # Larger tick font
plt.yticks(fontsize=28)  # Larger tick font
plt.legend(fontsize=28)
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig("Plot/All_SIR.png", dpi=500, bbox_inches="tight")

