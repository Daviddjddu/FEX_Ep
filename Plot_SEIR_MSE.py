### MSE with CI
import numpy as np
import matplotlib.pyplot as plt

# Load the MSE files
mse_file_1 = np.load('Data/mse_Fex_SEIR.npy')
mse_file_2 = np.load('Data/mse_RNN_SEIR.npy')
mse_file_3 = np.load('Data/mse_NN_SEIR.npy')
conf_interval_1 = np.load('Data/modified_mse_FEX_SEIR.npy')
conf_interval_2 = np.load('Data/modified_mse_RNN_SEIR.npy')
conf_interval_3 = np.load('Data/modified_mse_NN_SEIR.npy')

# Plot each MSE file on the same graph with different colors and specific labels
plt.figure(figsize=(12, 8))

# Plot first MSE file with specific label
plt.plot(range(250), mse_file_1, label='FEX MSE', color='blue', linestyle='--', linewidth=1)
plt.fill_between(range(250), mse_file_1 - conf_interval_1, mse_file_1 + conf_interval_1,
                 color='lightblue', alpha=0.95)

# Plot second MSE file with specific label
plt.plot(mse_file_2, label='RNN MSE', color='green', linestyle='-', linewidth=1)
plt.fill_between(range(250), mse_file_2 - conf_interval_2, mse_file_2 + conf_interval_2,
                 color='lightgreen', alpha=0.95)

# Plot third MSE file with specific label
plt.plot(mse_file_3, label='NN MSE', color='red', linestyle='-.', linewidth=1)
plt.fill_between(range(250), mse_file_3 - conf_interval_3, mse_file_3 + conf_interval_3,
                 color='lightpink', alpha=0.95)

# Add title and labels
plt.xlabel('Time Steps', fontsize=28)
plt.ylabel('MSE', fontsize=28)
plt.xticks(fontsize=28)  # Larger tick font
plt.yticks(fontsize=28)  # Larger tick font
plt.grid(True, alpha=0.3)

# Set y-axis to logarithmic scale
plt.yscale('log')

# Add a legend to label each line
plt.legend()

# Add a legend to label each line
plt.legend(loc='lower right', fontsize=28)
plt.tight_layout()  # Avoid clipping of labels
plt.savefig('Plot/MSE_SEIR_250.png', dpi=500, bbox_inches="tight")