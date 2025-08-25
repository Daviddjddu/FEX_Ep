import numpy as np
import matplotlib.pyplot as plt

# Load the MSE data
file_path_1 = 'Data/sir_training_data.npz'
file_path_2 = 'Data/sir_testing_data.npz'
data_1 = np.load(file_path_1)['data']
data_2 = np.load(file_path_2)['data']

# Simulation parameters
dt = 0.2
num_batches, num_time_steps, num_vars = data_1.shape
k = num_time_steps - 1  # Number of time steps

# Extract initial values
X0_batches_1 = data_1[:, 0, :]
X0_batches_2 = data_2[:, 0, :]

def compute_delta_X(X_current):
    X0, X1, X2= X_current

    # Equation for dS/dt
    delta_X0 = (
        (((((-0.6748*(X0)+0.2348*(X1)+0.2348*(X2)+-0.2348))*((0.0000*(X0)+0.9896*(X1)+0.0000*(X2)+0.3298))))+((0.0800*(0)+0.1664*(0)+-0.2279*(0)+0.3000)))
    )

    # Equation for dI/dt
    delta_X1 = (
        (((((0.2174*(X0)-0.8654*(X1)+0.2174*(X2)-0.2174))*((-0.4741*(1)-0.2465*(1)-0.3483*(1)-0.1354))))*((0.2901*(X0)-0.4001*(X1)-0.4001*(X2)+0.0167)))
    )

    # Equation for dR/dt
    delta_X2 = (
        (-0.5983*(((0.0211*(X0)+0.6839*(X1)-0.9730*(X2)+0.0060))*((-0.5044)))-0.0082)
    )
    
    return np.array([delta_X0, delta_X1, delta_X2])


def recursive_update(X0, steps):
    X = np.zeros((steps + 1, num_vars))
    X[0] = X0

    for step in range(1, steps + 1):
        delta_X = compute_delta_X(X[step - 1])
        X[step] = X[step - 1] + dt * delta_X

    return X

# Perform the recursive update for each batch
simulated_data_1 = np.zeros((num_batches, k + 1, num_vars))
simulated_data_2 = np.zeros((num_batches, k + 1, num_vars))

for i in range(num_batches):
    simulated_data_1[i] = recursive_update(X0_batches_1[i], k)
    simulated_data_2[i] = recursive_update(X0_batches_2[i], k)
    
# Compute the MSE curves for all batches
mse_curves_1 = np.mean((simulated_data_1 - data_1) ** 2, axis=2)  # Shape: (100, k+1)
mse_curves_2 = np.mean((simulated_data_2 - data_2) ** 2, axis=2)  # Shape: (100, k+1)

# Compute the mean and standard deviation of the MSE curves
mean_mse_curve_1 = np.mean(mse_curves_1, axis=0)
std_mse_curve_1 = np.std(mse_curves_1, axis=0)

mean_mse_curve_2 = np.mean(mse_curves_2, axis=0)
std_mse_curve_2 = np.std(mse_curves_2, axis=0)

# Compute 95% confidence interval
conf_interval_1 = 1.96 * (std_mse_curve_1 / np.sqrt(num_batches))
conf_interval_2 = 1.96 * (std_mse_curve_2 / np.sqrt(num_batches))

plt.figure(figsize=(12, 8))
# Plot mean MSE curve in bold
plt.plot(range(k + 1), mean_mse_curve_1, color='blue', linestyle = '-.', linewidth=5, label='MSE Training')
plt.plot(range(k + 1), mean_mse_curve_2, color='red',  linewidth= 5, label='MSE Testing')

# Plot confidence interval as a shaded region
plt.fill_between(range(k + 1), mean_mse_curve_1 - conf_interval_1, mean_mse_curve_1 + conf_interval_1, color='lightblue', alpha=0.4, label='95% CI Training')
plt.fill_between(range(k + 1), mean_mse_curve_2 - conf_interval_2, mean_mse_curve_2 + conf_interval_2, color='pink', alpha=0.4, label='95% CI Testing')

# Labels and title
plt.xlabel('Time Step', fontsize = 28)
plt.ylabel('MSE', fontsize = 28)
plt.legend(fontsize = 20)
plt.grid(True, alpha = 0.3)
plt.xticks(fontsize=28)  # Larger tick font
plt.yticks(fontsize=28)  # Larger tick font
ax = plt.gca()  # Get the current axis
ax.yaxis.get_offset_text().set_fontsize(28)
plt.savefig("Plot/Modified_SIR_MSE_TT.png", dpi=500, bbox_inches="tight")
