import numpy as np
import matplotlib.pyplot as plt

# Load the MSE data
file_path = 'Data/seird_testing_data.npz'
data = np.load(file_path)['data']

# Simulation parameters
dt = 0.2
num_batches, num_time_steps, num_vars = data.shape
k = num_time_steps - 1  # Number of time steps

# Extract initial values
X0_batches = data[:, 0, :]

def compute_delta_X(X_current):
    X0, X1, X2, X3, X4 = X_current

    # Equation for dS/dt
    delta_X0 = (
        (((((0.9762*(X0)+-0.0526*(X1)+-0.0526*(X2)+-0.0526*(X3)+-0.0526*(X4)+0.0094))*((0.1118*(X0)+0.1118*(X1)+-0.7630*(X2)+0.1118*(X3)+0.1118*(X4)+-0.0735))))+((-0.0714*(X0)+-0.0320*(X1)+-0.0698*(X2)+-0.0320*(X3)+-0.0320*(X4)+0.0336)))
    )

    # Equation for dE/dt
    delta_X1 = (
        # (((((0.8318*(X0)+-0.0068*(X1)+-0.0053*(X2)+-0.0085*(X3)+-0.0076*(X4)+-0.0315))*((-0.1778*(X0)+-0.2013*(X1)+0.8948*(X2)+-0.1849*(X3)+-0.1832*(X4)+0.1381))))+((0.0398*(np.sin(X0))+-0.5063*(np.sin(X1))+0.0442*(np.sin(X2))+0.0005*(np.sin(X3))+0.0004*(np.sin(X4))+-0.0023)))
        (((((-0.1079*(X0)+-0.1185*(X1)+0.6853*(X2)+-0.1165*(X3)+-0.1157*(X4)+0.1117))*((1.1483*(np.sin(X0))+0.0120*(np.sin(X1))+0.0113*(np.sin(X2))+0.0125*(np.sin(X3))+0.0113*(np.sin(X4))+-0.1651))))-((0.0408*(X0)+0.5459*(X1)+-0.0775*(X2)+0.0452*(X3)+0.0451*(X4)+-0.0445)))
    )

    # Equation for dI/dt
    delta_X2 = (
        (0.1551*(((-0.0666*(X0)+3.1575*(X1)+-1.6787*(X2)+-0.0666*(X3)+-0.0666*(X4)+0.0414))-((0.1167*(1)+-0.0271*(1)+0.0042*(1)+-0.0808*(1)+-0.0137*(1)+-0.0501)))+-0.0040)
    )

    # Equation for dR/dt
    delta_X3 = (
        (-0.0922*(((0.0972*(1)+-0.1409*(1)+-0.0568*(1)+0.0889*(1)+-0.1079*(1)+-0.0697))-((-0.1972*(X0)+-0.1972*(X1)+1.9712*(X2)+-0.1972*(X3)+-0.1972*(X4)+-0.0872)))+0.0088)
    )

    # Equation for dD/dt
    delta_X4 = (
        (-0.1503*(((0.0342*(0)+0.0113*(0)+-0.0922*(0)+-0.1008*(0)+-0.0577*(0)+-0.2195))*((0.0242*(X0)+0.0242*(X1)+1.5394*(X2)+0.0242*(X3)+0.0242*(X4)+0.0036)))+-0.0009)
    )
    return np.array([delta_X0, delta_X1, delta_X2, delta_X3, delta_X4])

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

# Compute the mean and standard deviation of the MSE curves
mean_mse_curve = np.mean(mse_curves, axis=0)
std_mse_curve = np.std(mse_curves, axis=0)

# Compute 95% confidence interval
conf_interval = 1.96 * (std_mse_curve / np.sqrt(num_batches))

np.save('Data/mse_Fex_SEIRD.npy', mean_mse_curve)
np.save("Data/modified_mse_FEX_SEIRD.npy", conf_interval)

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
plt.title('MSE Curves for All Batches with Mean and Confidence Interval (SEIRD)')
plt.xlabel('Time Step')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha = 0.2)
plt.savefig("Plot/Modified_SEIRD_MSE.png", dpi=300, bbox_inches="tight")

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 750 time steps, 4 variables)

# dt = 0.2

# # Extract initial values (first time step for each batch)
# X0_batches = data[:, 0, :]  # Shape: (100, 4)

# def compute_delta_X(X_current):
#     X0, X1, X2, X3, X4 = X_current

#     # Equation for dS/dt
#     delta_X0 = (
#         (((((0.9762*(X0)+-0.0526*(X1)+-0.0526*(X2)+-0.0526*(X3)+-0.0526*(X4)+0.0094))*((0.1118*(X0)+0.1118*(X1)+-0.7630*(X2)+0.1118*(X3)+0.1118*(X4)+-0.0735))))+((-0.0714*(X0)+-0.0320*(X1)+-0.0698*(X2)+-0.0320*(X3)+-0.0320*(X4)+0.0336)))
#     )

#     # Equation for dE/dt
#     delta_X1 = (
#         # (((((0.8318*(X0)+-0.0068*(X1)+-0.0053*(X2)+-0.0085*(X3)+-0.0076*(X4)+-0.0315))*((-0.1778*(X0)+-0.2013*(X1)+0.8948*(X2)+-0.1849*(X3)+-0.1832*(X4)+0.1381))))+((0.0398*(np.sin(X0))+-0.5063*(np.sin(X1))+0.0442*(np.sin(X2))+0.0005*(np.sin(X3))+0.0004*(np.sin(X4))+-0.0023)))
#         (((((-0.1079*(X0)+-0.1185*(X1)+0.6853*(X2)+-0.1165*(X3)+-0.1157*(X4)+0.1117))*((1.1483*(np.sin(X0))+0.0120*(np.sin(X1))+0.0113*(np.sin(X2))+0.0125*(np.sin(X3))+0.0113*(np.sin(X4))+-0.1651))))-((0.0408*(X0)+0.5459*(X1)+-0.0775*(X2)+0.0452*(X3)+0.0451*(X4)+-0.0445)))
#     )

#     # Equation for dI/dt
#     delta_X2 = (
#         (0.1551*(((-0.0666*(X0)+3.1575*(X1)+-1.6787*(X2)+-0.0666*(X3)+-0.0666*(X4)+0.0414))-((0.1167*(1)+-0.0271*(1)+0.0042*(1)+-0.0808*(1)+-0.0137*(1)+-0.0501)))+-0.0040)
#     )

#     # Equation for dR/dt
#     delta_X3 = (
#         (-0.0922*(((0.0972*(1)+-0.1409*(1)+-0.0568*(1)+0.0889*(1)+-0.1079*(1)+-0.0697))-((-0.1972*(X0)+-0.1972*(X1)+1.9712*(X2)+-0.1972*(X3)+-0.1972*(X4)+-0.0872)))+0.0088)
#     )

#     # Equation for dD/dt
#     delta_X4 = (
#         (-0.1503*(((0.0342*(0)+0.0113*(0)+-0.0922*(0)+-0.1008*(0)+-0.0577*(0)+-0.2195))*((0.0242*(X0)+0.0242*(X1)+1.5394*(X2)+0.0242*(X3)+0.0242*(X4)+0.0036)))+-0.0009)
#     )
#     return np.array([delta_X0, delta_X1, delta_X2, delta_X3, delta_X4])

# def recursive_update(X0, steps):
#     X = np.zeros((steps + 1, 5))  # We now have 4 variables
#     X[0] = X0

#     for step in range(1, steps + 1):
#         delta_X = compute_delta_X(X[step - 1])
#         X[step] = X[step - 1] + 0.2 * delta_X

#     return X

# # Number of time steps (k)
# k = 249

# # Perform the recursive update for each batch
# simulated_data = np.zeros((100, k + 1, 5))  # To store data for all batches
# for i in range(100):
#     simulated_data[i] = recursive_update(X0_batches[i], k)

# # Calculate Mean Squared Error (MSE) for each time step
# mse_per_time_step = np.zeros(k + 1)
# for t in range(k + 1):
#     # Compute the MSE for all batches and variables at time step t
#     mse_per_time_step[t] = np.mean((simulated_data[:, t, :] - data[:, t, :])**2)
# np.save('mse_Fex_SEIRD.npy', mse_per_time_step)

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

# #Only consider the first dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 749  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X0(X0, X1, X2, X3, X4):
#     # Equation for dS/dt
#     delta_X0 = (
#     (((((-0.7370*(X0)+0.0255*(X1)-0.0424*(X2)-0.0011*(X3)-0.0161*(X4)-0.0154))-((0.7212*(X0)+0.0552*(X1)-0.0127*(X2)+0.0286*(X3)+0.0136*(X4)-0.0451))))*((0.0207*(X0)+0.0207*(X1)+0.6508*(X2)+0.0207*(X3)+0.0207*(X4)-0.0207)))
#     )
#     return delta_X0

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X0_initial = data[batch_index, 0, 0]  
#     X1_data = data[batch_index, :, 1]  # Actual data for I
#     X3_data = data[batch_index, :, 3]  # Actual data for R
#     X2_data = data[batch_index, :, 2]  # Actual data for R
#     X4_data = data[batch_index, :, 4]    

#     # Simulate the value for X0
#     simulated_X0 = np.zeros(k + 1)
#     simulated_X0[0] = X0_initial

#     for t in range(1, k + 1):
#         delta_X0 = compute_delta_X0(simulated_X0[t - 1],X1_data[t - 1],  X2_data[t - 1], X3_data[t - 1], X4_data[t-1])
#         simulated_X0[t] = simulated_X0[t - 1] + 0.2 * delta_X0  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X0 - data[batch_index, :, 0])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k+1), simulated_X0, label = 'simulated', color  = 'red', linestyle = '--')
# # plt.plot(range(k + 1), data[batch_index, :, 0] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X0 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()
# print(simulated_X0[:15])
# print(data[99, :15, 0])


# #Only consider the Second dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 749  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X1(X0, X1, X2, X3, X4):
#     # Equation for dS/dt
#     delta_X1 = (
#     (((((-0.5701*(np.sin(X0))+0.0716*(np.sin(X1))+0.0580*(np.sin(X2))+0.0665*(np.sin(X3))+0.0623*(np.sin(X4))-0.0237))*((-0.0307*(np.sin(X0))-0.0169*(np.sin(X1))-1.4890*(np.sin(X2))-0.0298*(np.sin(X3))-0.0229*(np.sin(X4))+0.0533))))-((-0.0148*(X0)+0.5004*(X1)-0.0527*(X2)+0.0018*(X3)+0.0019*(X4)-0.0008)))
#     )
#     return delta_X1

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X1_initial = data[batch_index, 0, 1]  
#     X0_data = data[batch_index, :, 0]  # Actual data for I
#     X3_data = data[batch_index, :, 3]  # Actual data for R
#     X2_data = data[batch_index, :, 2]  # Actual data for R
#     X4_data = data[batch_index, :, 4]    

#     # Simulate the value for X1
#     simulated_X1 = np.zeros(k + 1)
#     simulated_X1[0] = X1_initial

#     for t in range(1, k + 1):
#         delta_X1 = compute_delta_X1(X0_data[t - 1], simulated_X1[t - 1], X2_data[t - 1], X3_data[t - 1], X4_data[t-1])
#         simulated_X1[t] = simulated_X1[t - 1] + 0.2 * delta_X1  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X1 - data[batch_index, :, 1])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k+1), simulated_X1, label = 'simulated', color  = 'red', linestyle = '--')
# # plt.plot(range(k + 1), data[batch_index, :, 1] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X1 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# #Only consider the Third dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 749  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X2(X0, X1, X2, X3, X4):
#     # Equation for dS/dt
#     delta_X2 = (
#     (-0.1599*(((-0.0434))+((-0.0374*(X0)-3.1652*(X1)+1.5266*(X2)-0.0374*(X3)-0.0374*(X4)-0.1070)))-0.0300)
#     )
#     return delta_X2

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X2_initial = data[batch_index, 0, 2]  
#     X0_data = data[batch_index, :, 0]  # Actual data for I
#     X3_data = data[batch_index, :, 3]  # Actual data for R
#     X1_data = data[batch_index, :, 1]  # Actual data for R
#     X4_data = data[batch_index, :, 4]    

#     # Simulate the value for X1
#     simulated_X2 = np.zeros(k + 1)
#     simulated_X2[0] = X2_initial

#     for t in range(1, k + 1):
#         delta_X2 = compute_delta_X2(X0_data[t - 1], X1_data[t - 1], simulated_X2[t - 1], X3_data[t - 1], X4_data[t-1])
#         simulated_X2[t] = simulated_X2[t - 1] + 0.2 * delta_X2  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X2 - data[batch_index, :, 2])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# #plt.plot(range(k+1), simulated_X2, label = 'simulated', color  = 'red', linestyle = '--')
# #plt.plot(range(k + 1), data[batch_index, :, 2] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X2 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# #Only consider the Fourth dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 749  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X3(X0, X1, X2, X3, X4):
#     # Equation for dS/dt
#     delta_X3 = (
#     (0.1150*(((-0.1039))+((-0.0764*(X0)-0.0764*(X1)+1.6620*(X2)-0.0764*(X3)+-0.0764*(X4)+0.0572)))+0.0142)
#     )
#     return delta_X3

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X3_initial = data[batch_index, 0, 3]  
#     X0_data = data[batch_index, :, 0]  # Actual data for I
#     X1_data = data[batch_index, :, 1]  # Actual data for R
#     X2_data = data[batch_index, :, 2]  # Actual data for R
#     X4_data = data[batch_index, :, 4]    

#     # Simulate the value for X3
#     simulated_X3 = np.zeros(k + 1)
#     simulated_X3[0] = X3_initial

#     for t in range(1, k + 1):
#         delta_X3 = compute_delta_X3(X0_data[t - 1], X1_data[t - 1], X2_data[t - 1], simulated_X3[t - 1], X4_data[t-1])
#         simulated_X3[t] = simulated_X3[t - 1] + 0.2 * delta_X3  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X3 - data[batch_index, :, 3])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# #plt.plot(range(k+1), simulated_X3, label = 'simulated', color  = 'red', linestyle = '--')
# #plt.plot(range(k + 1), data[batch_index, :, 3] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X3 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# #Only consider the Fifth dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 149  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X4(X0, X1, X2, X3, X4):
#     # Equation for dS/dt
#     delta_X4 = (
#    (0.0320*(((-0.0760*(X0)-0.0760*(X1)+1.4877*(X2)-0.0760*(X3)-0.0760*(X4)+0.0790))+((-0.1152*(1)-0.1657*(1)+0.0633*(1)+0.0520*(1)+0.0650*(1)+0.0687)))+0.0009)
#     )
#     return delta_X4

# # Initialize to store residual squares for all batches
# residual_squares_all_batches = np.zeros((num_batches, k + 1))

# # Loop through each batch
# for batch_index in range(num_batches):
#     # Extract initial value for the first variable (S)
#     X4_initial = data[batch_index, 0, 4]  
#     X0_data = data[batch_index, :, 0]  # Actual data for I
#     X3_data = data[batch_index, :, 3]  # Actual data for R
#     X2_data = data[batch_index, :, 2]  # Actual data for R
#     X1_data = data[batch_index, :, 1]    

#     # Simulate the value for X4
#     simulated_X4 = np.zeros(k + 1)
#     simulated_X4[0] = X4_initial

#     for t in range(1, k + 1):
#         delta_X4 = compute_delta_X4(X0_data[t - 1], X1_data[t-1], X2_data[t - 1], X3_data[t - 1], simulated_X4[t - 1] )
#         simulated_X4[t] = simulated_X4[t - 1] + 0.2 * delta_X4  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X4 - data[batch_index, :k+1, 4])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# #plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# plt.plot(range(k+1), mean_residual_squares, label = 'simulated', color  = 'red', linestyle = '--')
# # plt.plot(range(k + 1), data[batch_index, :, 4] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X4 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()