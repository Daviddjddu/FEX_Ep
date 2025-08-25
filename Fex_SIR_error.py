import numpy as np
import time
import matplotlib.pyplot as plt
    
# Load the MSE data
file_path = 'Data/sir_testing_data.npz'
data = np.load(file_path)['data']

# Simulation parameters
dt = 0.2
num_batches, num_time_steps, num_vars = data.shape
k = num_time_steps - 1  # Number of time steps

# Extract initial values
X0_batches = data[:, 0, :]

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
simulated_data = np.zeros((num_batches, k + 1, num_vars))

for i in range(num_batches):
    simulated_data[i] = recursive_update(X0_batches[i], k)

# Compute the MSE curves for all batches
mse_curves = np.mean((simulated_data - data) ** 2, axis=2)  # Shape: (100, k+1) 
mean_mse_curve = np.mean(mse_curves, axis=0)
std_mse_curve = np.std(mse_curves, axis=0)

# Compute 95% confidence interval
conf_interval = 1.96 * (std_mse_curve / np.sqrt(num_batches))

np.save('Data/mse_Fex_SIR.npy', mean_mse_curve)
np.save("Data/modified_mse_FEX_SIR.npy", conf_interval)

# Plot all individual MSE curves
plt.figure(figsize=(12, 6))
for i in range(num_batches):
    plt.plot(range(k + 1), mse_curves[i], color='lightgreen', alpha=0.3)

# Plot mean MSE curve in bold
plt.plot(range(k + 1), mean_mse_curve, color='blue', linewidth=2.5, label='Mean MSE')

# Plot confidence interval as a shaded region
plt.fill_between(range(k + 1), mean_mse_curve - conf_interval, mean_mse_curve + conf_interval,
                 color='red', alpha=0.95, label='95% CI')

# Labels and title
plt.title('MSE Curves for All Batches with Mean and Confidence Interval (SIR)')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha = 0.2)
plt.savefig("Plot/Modified_SIR_MSE.png", dpi=300, bbox_inches="tight")


# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'sir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 750 time steps, 3 variables)

# dt = 0.2

# # Extract initial values (first time step for each batch)
# X0_batches = data[:, 0, :]  # Shape: (100, 3)

# def compute_delta_X(X_current):
#     X0, X1, X2= X_current

#     # Equation for dS/dt
#     delta_X0 = (
#         (((((-0.6748*(X0)+0.2348*(X1)+0.2348*(X2)+-0.2348))*((0.0000*(X0)+0.9896*(X1)+0.0000*(X2)+0.3298))))+((0.0800*(0)+0.1664*(0)+-0.2279*(0)+0.3000)))
#     )

#     # Equation for dI/dt
#     delta_X1 = (
#         (((((0.2174*(X0)-0.8654*(X1)+0.2174*(X2)-0.2174))*((-0.4741*(1)-0.2465*(1)-0.3483*(1)-0.1354))))*((0.2901*(X0)-0.4001*(X1)-0.4001*(X2)+0.0167)))
#     )

#     # Equation for dR/dt
#     delta_X2 = (
#         (-0.5983*(((0.0211*(X0)+0.6839*(X1)-0.9730*(X2)+0.0060))*((-0.5044)))-0.0082)
#     )
    
#     return np.array([delta_X0, delta_X1, delta_X2])

# def recursive_update(X0, steps):
#     X = np.zeros((steps + 1, 3))  # We now have 3 variables
#     X[0] = X0

#     for step in range(1, steps + 1):
#         delta_X = compute_delta_X(X[step - 1])
#         X[step] = X[step - 1] + dt * delta_X
        
#         #  # Enforce the condition X0 + X1 + X2 = 1 by adjusting X0
#         # X1 = X[step][1]
#         # X2 = X[step][2]
#         # X[step][0] = 1 - (X1 + X2)  # Adjust X0 to satisfy the constraint

#     return X

# # Number of time steps (k)
# k = 249  

# # Perform the recursive update for each batch
# simulated_data = np.zeros((100, k + 1, 3))  # To store data for all batches
# for i in range(100):
#     simulated_data[i] = recursive_update(X0_batches[i], k)

# # Calculate Mean Squared Error (MSE) for each time step
# mse_per_time_step = np.zeros(k + 1)
# for t in range(k + 1):
#     # Compute the MSE for all batches and variables at time step t
#     mse_per_time_step[t] = np.mean((simulated_data[:, t, :] - data[:, t, :])**2)
# np.save('mse_Fex_SIR.npy', mse_per_time_step)

# # Calculate the mean MSE across all time steps
# mean_mse = np.mean(mse_per_time_step)

# print(f"Mean MSE across all batches: {mean_mse}")

# # Plot MSE for each time step
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mse_per_time_step, label='MSE per Time Step', color='blue', linestyle='--')
# plt.title('MSE per Time Step')
# plt.xlabel('Time Step')
# plt.ylabel('MSE')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Only consider the first dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'sir_testing_data.npz'
# data = np.load(file_path)['data']

# # Constants
# num_batches = data.shape[0]
# num_time_steps = data.shape[1]
# k = num_time_steps - 1  # Number of time steps (assuming from the given shape)

# def compute_delta_X0(X0, X1, X2):
#     # Equation for delta_X2 using given formula
#     delta_X0 = (
#       (((((0.0657*(X0)+0.9274*(X1)+0.0739*(X2)-0.0394))*((-0.4422*(X0)+0.5103*(X1)+0.5332*(X2)+0.0332))))+((-0.1063*(np.sin(X0))-0.3238*(np.sin(X1))+0.2016*(np.sin(X2))+0.1010)))
#     )
#     return delta_X0

# # Initialize array to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_idx in range(num_batches):
#     # Extract the initial value for X1 and the data for X0 and X2
#     X0_initial = data[batch_idx, 0, 0]
#     X1_data = data[batch_idx, :, 1]  # Actual data for X0 (Susceptible)
#     X2_data = data[batch_idx, :, 2]  # Actual data for X2 (Infectious)

#     # Simulate the values for X1
#     simulated_X0 = np.zeros(k + 1)
#     simulated_X0[0] = X0_initial

#     for t in range(1, k + 1):
#         delta_X0 = compute_delta_X0(simulated_X0[t-1], X1_data[t-1], X2_data[t-1])
#         simulated_X0[t] = simulated_X0[t-1] + 0.2 * delta_X0  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_idx, :] = (simulated_X0 - data[batch_idx, :, 0])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares_X0 = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares_X0, label='Mean Residual Squares', color='red', linestyle='--')
# plt.title('Mean Residual Squares for X0 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Optionally print overall mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares_X0)
# print(f"Overall Mean Residual Square Error for X1 over all time steps and batches: {overall_mean_residual_square:.5f}")


# # Only consider the second dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'sir_testing_data.npz'
# data = np.load(file_path)['data']

# # Constants
# num_batches = data.shape[0]
# num_time_steps = data.shape[1]
# k = num_time_steps - 1  # Number of time steps (assuming from the given shape)

# def compute_delta_X1(X0, X1, X2):
#     # Equation for delta_X2 using given formula
#     delta_X1 = (
#       (((((0.2174*(X0)-0.8654*(X1)+0.2174*(X2)-0.2174))*((-0.4741*(1)-0.2465*(1)-0.3483*(1)-0.1354))))*((0.2901*(X0)-0.4001*(X1)-0.4001*(X2)+0.0167)))
#     )
#     return delta_X1

# # Initialize array to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_idx in range(num_batches):
#     # Extract the initial value for X1 and the data for X0 and X2
#     X1_initial = data[batch_idx, 0, 1]
#     X0_data = data[batch_idx, :, 0]  # Actual data for X0 (Susceptible)
#     X2_data = data[batch_idx, :, 2]  # Actual data for X2 (Infectious)

#     # Simulate the values for X1
#     simulated_X1 = np.zeros(k + 1)
#     simulated_X1[0] = X1_initial

#     for t in range(1, k + 1):
#         delta_X1 = compute_delta_X1(X0_data[t - 1], simulated_X1[t - 1], X2_data[t - 1])
#         simulated_X1[t] = simulated_X1[t - 1] + 0.2 * delta_X1  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_idx, :] = (simulated_X1 - data[batch_idx, :, 1])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares, label='Mean Residual Squares', color='red', linestyle='--')
# plt.title('Mean Residual Squares for X1 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Optionally print overall mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares)
# print(f"Overall Mean Residual Square Error for X1 over all time steps and batches: {overall_mean_residual_square:.5f}")

# #Only consider the Third dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'sir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 249  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X2(X0, X1, X2):
#     # Equation for dS/dt
#     delta_X2 = (
#     (-0.5983*(((0.0211*(X0)+0.6839*(X1)-0.9730*(X2)+0.0060))*((-0.5044)))-0.0082)
#     )
#     return delta_X2

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X2_initial = data[batch_index, 0, 2]  
#     X0_data = data[batch_index, :, 0]  # Actual data for I
#     X1_data = data[batch_index, :, 1]  # Actual data for R

#     # Simulate the value for X0
#     simulated_X2 = np.zeros(k + 1)
#     simulated_X2[0] = X2_initial

#     for t in range(1, k + 1):
#         delta_X2 = compute_delta_X2(X0_data[t - 1],  X1_data[t - 1], simulated_X2[t - 1])
#         simulated_X2[t] = simulated_X2[t - 1] + 0.2 * delta_X2  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X2 - data[batch_index, :, 2])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k + 1), data[batch_index, :, 2] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X2 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Print mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares)
# print(f"Overall Mean Residual Square Error for X2 over all time steps and batches: {overall_mean_residual_square}")