import numpy as np
import matplotlib.pyplot as plt

# Load the MSE data
file_path = 'Data/seir_testing_data.npz'
data = np.load(file_path)['data']

# Simulation parameters
dt = 0.2
num_batches, num_time_steps, num_vars = data.shape
k = num_time_steps - 1  # Number of time steps

# Extract initial values
X0_batches = data[:, 0, :]

def compute_delta_X(X_current):
    X0, X1, X2, X3 = X_current

    # Equation for dS/dt
    delta_X0 = (
         (((((-0.0558*(X0)+-0.0558*(X1)+-1.0160*(X2)+-0.0558*(X3)+-0.0484))*((0.3527*(X0)+-0.5847*(X1)+-0.5847*(X2)+-0.5847*(X3)+0.0900))))-((0.1865*(X0)+-0.2158*(X1)+0.2591*(X2)+-0.2158*(X3)+-0.0326)))
        # 0.3-0.5*X0-0.9*X0*X2
    )   

    # Equation for dE/dt
    delta_X1 = (
        # (((((0.0133*(np.sin(X0))+0.0248*(np.sin(X1))+-1.3636*(np.sin(X2))+0.0191*(np.sin(X3))+-0.1553))*((-0.3990*(X0)+0.2668*(X1)+0.2679*(X2)+0.2699*(X3)+-0.0476))))+((-0.1048*(X0)+-0.9118*(X1)+0.2864*(X2)+-0.0105*(X3)+0.0419))) 
        (((((0.4636*(np.sin(X0))+-0.2237*(np.sin(X1))+-0.2442*(np.sin(X2))+-0.2263*(np.sin(X3))+-0.0455))*((0.0038*(np.sin(X0))+-0.0254*(np.sin(X1))+1.3707*(np.sin(X2))+-0.0265*(np.sin(X3))+0.0398))))+((-0.0523*(X0)+-0.9371*(X1)+0.3319*(X2)+-0.0374*(X3)+0.0423)))     
        # (((((-0.0703*(X0)+-0.0703*(X1)+-1.5583*(X2)+-0.0703*(X3)+0.1012))*((-0.3077*(X0)+0.2971*(X1)+0.2971*(X2)+0.2971*(X3)+0.0390))))+((0.0733*(X0)+-0.8454*(X1)+0.5548*(X2)+0.0546*(X3)+-0.0649)))  
        # 0.9*(X0)*(X2)-0.9*(X1)
        # 0.8999*(X0)*(X2)-0.9*(X1)+ 0.0001*(X2)+ 0.0001
    )

    # Equation for dI/dt
    delta_X2 = (
        (-0.2979*(((0.0347*(X0)+1.6549*(X1)-1.3155*(X2)+0.0347*(X3)-0.0212))*((-0.1810*(1)-0.0803*(1)-0.2554*(1)-0.3325*(1)-0.3939)))-0.0050) 
        # 0.6*X1-0.5*X2
    )

    # Equation for dR/dt
    delta_X3 = (
        (0.2761*(((0.5029*(X0)+0.1183*(X1)+0.3777*(X2)-0.4662*(X3)-0.1053))-((-0.3216*(X0)+0.0182*(X1)-0.4468*(X2)+0.5203*(X3)-0.1654)))-0.0442) 
        # 0.2*(X2) - 0.3*(X3) + 0.2*(X0)
    )

    return np.array([delta_X0, delta_X1, delta_X2, delta_X3])

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

np.save('Data/mse_Fex_SEIR.npy', mean_mse_curve)
np.save("Data/modified_mse_FEX_SEIR.npy", conf_interval)

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
plt.savefig("Plot/Modified_SEIR_MSE.png", dpi=300, bbox_inches="tight")

# std_mse_curve = np.std(mse_curves, axis=0)

# # Compute 95% confidence interval
# conf_interval = 1.96 * (std_mse_curve / np.sqrt(num_batches))

# np.save("modified_mse_FEX_SEIR.npy", conf_interval)

# # Plot all individual MSE curves
# plt.figure(figsize=(12, 6))
# for i in range(num_batches):
#     plt.plot(range(k + 1), mse_curves[i], color='lightgreen', alpha=0.3)

# # Plot mean MSE curve in bold
# plt.plot(range(k + 1), mean_mse_curve, color='blue', linewidth=2.5, label='Mean MSE')

# # Plot confidence interval as a shaded region
# plt.fill_between(range(k + 1), mean_mse_curve - conf_interval, mean_mse_curve + conf_interval,
#                  color='red', alpha=0.95, label='95% CI')

# # Labels and title
# plt.title('MSE Curves for All Batches with Mean and Confidence Interval (SEIR)')
# plt.xlabel('Time Step')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid(True, alpha = 0.2)
# plt.savefig("Modified_SEIR_MSE.png", dpi=300, bbox_inches="tight")

# # Display the plot
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 150 time steps, 4 variables)

# dt = 0.2

# # Extract initial values (first time step for each batch)
# X0_batches = data[:, 0, :]  # Shape: (100, 4)

# def compute_delta_X(X_current):
#     X0, X1, X2, X3 = X_current

#     # Equation for dS/dt
#     delta_X0 = (
#          (((((-0.0558*(X0)+-0.0558*(X1)+-1.0160*(X2)+-0.0558*(X3)+-0.0484))*((0.3527*(X0)+-0.5847*(X1)+-0.5847*(X2)+-0.5847*(X3)+0.0900))))-((0.1865*(X0)+-0.2158*(X1)+0.2591*(X2)+-0.2158*(X3)+-0.0326)))
#         # 0.3-0.5*X0-0.9*X0*X2
#     )   

#     # Equation for dE/dt
#     delta_X1 = (
#         # (((((0.0133*(np.sin(X0))+0.0248*(np.sin(X1))+-1.3636*(np.sin(X2))+0.0191*(np.sin(X3))+-0.1553))*((-0.3990*(X0)+0.2668*(X1)+0.2679*(X2)+0.2699*(X3)+-0.0476))))+((-0.1048*(X0)+-0.9118*(X1)+0.2864*(X2)+-0.0105*(X3)+0.0419))) 
#         (((((0.4636*(np.sin(X0))+-0.2237*(np.sin(X1))+-0.2442*(np.sin(X2))+-0.2263*(np.sin(X3))+-0.0455))*((0.0038*(np.sin(X0))+-0.0254*(np.sin(X1))+1.3707*(np.sin(X2))+-0.0265*(np.sin(X3))+0.0398))))+((-0.0523*(X0)+-0.9371*(X1)+0.3319*(X2)+-0.0374*(X3)+0.0423)))     
#         # (((((-0.0703*(X0)+-0.0703*(X1)+-1.5583*(X2)+-0.0703*(X3)+0.1012))*((-0.3077*(X0)+0.2971*(X1)+0.2971*(X2)+0.2971*(X3)+0.0390))))+((0.0733*(X0)+-0.8454*(X1)+0.5548*(X2)+0.0546*(X3)+-0.0649)))  
#         # 0.9*(X0)*(X2)-0.9*(X1)
#         # 0.8999*(X0)*(X2)-0.9*(X1)+ 0.0001*(X2)+ 0.0001
#     )

#     # Equation for dI/dt
#     delta_X2 = (
#         (-0.2979*(((0.0347*(X0)+1.6549*(X1)-1.3155*(X2)+0.0347*(X3)-0.0212))*((-0.1810*(1)-0.0803*(1)-0.2554*(1)-0.3325*(1)-0.3939)))-0.0050) 
#         # 0.6*X1-0.5*X2
#     )

#     # Equation for dR/dt
#     delta_X3 = (
#         (0.2761*(((0.5029*(X0)+0.1183*(X1)+0.3777*(X2)-0.4662*(X3)-0.1053))-((-0.3216*(X0)+0.0182*(X1)-0.4468*(X2)+0.5203*(X3)-0.1654)))-0.0442) 
#         # 0.2*(X2) - 0.3*(X3) + 0.2*(X0)
#     )
    
#     return np.array([delta_X0, delta_X1, delta_X2, delta_X3])

# def recursive_update(X0, steps):
#     X = np.zeros((steps + 1, 4))  # We now have 4 variables
#     X[0] = X0

#     for step in range(1, steps + 1):
#         delta_X = compute_delta_X(X[step - 1])
#         X[step] = X[step - 1] + dt * delta_X

#     return X

# # Number of time steps (k)
# k = 249 

# # Perform the recursive update for each batch
# simulated_data = np.zeros((100, k + 1, 4))  # To store data for all batches
# for i in range(100):
#     simulated_data[i] = recursive_update(X0_batches[i], k)

# # Calculate Mean Squared Error (MSE) for each time step
# mse_per_time_step = np.zeros(k + 1)
# for t in range(k + 1):
#     # Compute the MSE for all batches and variables at time step t
#     mse_per_time_step[t] = np.mean((simulated_data[:, t, :] - data[:, t, :])**2)
# np.save('mse_Fex_SEIR.npy', mse_per_time_step)

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

# print(data[0,0,0])
# print(simulated_data[0,0,0])


# #Only consider the first dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 249  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X0(X0, X1, X2, X3):
#     # Equation for dS/dt
#     delta_X0 = (
#       (((((0.0316*(X0)+0.0310*(X1)-1.5135*(X2)+0.0292*(X3)-0.3021))*((0.2036*(X0)-0.3793*(X1)-0.3788*(X2)-0.3786*(X3)-0.1016))))+((-0.0778*(X0)+0.2637*(X1)-0.4781*(X2)+0.2636*(X3)-0.0943)))
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

#     # Simulate the value for X0
#     simulated_X0 = np.zeros(k + 1)
#     simulated_X0[0] = X0_initial

#     for t in range(1, k + 1):
#         delta_X0 = compute_delta_X0(simulated_X0[t - 1],X1_data[t - 1],  X2_data[t - 1], X3_data[t - 1])
#         simulated_X0[t] = simulated_X0[t - 1] + 0.2 * delta_X0  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X0 - data[batch_index, :, 0])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k + 1), data[batch_index, :, 0] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X0 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Print mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares)
# print(f"Overall Mean Residual Square Error for X0 over all time steps and batches: {overall_mean_residual_square}")


# #Only consider the second dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 750 time steps, 3 variables)

# # Constants
# num_batches = 1 #data.shape[0]
# k = 249  # Number of time steps (assumed from 750 total steps)

# def compute_delta_X1(X0, X1, X2, X3):
#     # Equation for dS/dt
#     delta_X1 = (
#     (((((0.0590*(X0)+0.0718*(X1)-2.1506*(X2)+0.0707*(X3)+0.0136))*((-0.2222*(np.sin(X0))+0.1978*(np.sin(X1))+0.2049*(np.sin(X2))+0.1963*(np.sin(X3))+0.0340))))+((0.2262*(X0)-0.7082*(X1)+0.6999*(X2)+0.1902*(X3)-0.2106)))
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

#     # Simulate the value for X0
#     simulated_X1 = np.zeros(k + 1)
#     simulated_X1[0] = X1_initial

#     for t in range(1, k + 1):
#         delta_X1 = compute_delta_X1(X0_data[t - 1], simulated_X1[t - 1],  X2_data[t - 1], X3_data[t - 1])
#         simulated_X1[t] = simulated_X1[t - 1] + 0.2 * delta_X1  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X1 - data[batch_index, :, 1])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k + 1), data[batch_index, :, 1] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X1 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Print mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares)
# print(f"Overall Mean Residual Square Error for X1 over all time steps and batches: {overall_mean_residual_square}")

# #Only consider the Third dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 249  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X2(X0, X1, X2, X3):
#     # Equation for dS/dt
#     delta_X2 = (
#     (-0.2979*(((0.0347*(X0)+1.6549*(X1)-1.3155*(X2)+0.0347*(X3)-0.0212))*((-0.1810*(1)-0.0803*(1)-0.2554*(1)-0.3325*(1)-0.3939)))-0.0050)
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
#     X3_data = data[batch_index, :, 3]  # Actual data for R

#     # Simulate the value for X0
#     simulated_X2 = np.zeros(k + 1)
#     simulated_X2[0] = X2_initial

#     for t in range(1, k + 1):
#         delta_X2 = compute_delta_X2(X0_data[t - 1],  X1_data[t - 1], simulated_X2[t - 1], X3_data[t - 1])
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

# #Only consider the Fourth dimension of the MSE

# import numpy as np
# import matplotlib.pyplot as plt

# # Load the data from the .npz file
# file_path = 'seir_testing_data.npz'
# data = np.load(file_path)['data']
# # Shape of the data: (100 batches, 50 time steps, 3 variables)

# # Constants
# num_batches = data.shape[0]
# k = 249  # Number of time steps (assumed from 50 total steps)

# def compute_delta_X3(X0, X1, X2, X3):
#     # Equation for dS/dt
#     delta_X3 = (
#     (0.2761*(((0.5029*(X0)+0.1183*(X1)+0.3777*(X2)-0.4662*(X3)-0.1053))-((-0.3216*(X0)+0.0182*(X1)-0.4468*(X2)+0.5203*(X3)-0.1654)))-0.0442)
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
#     X2_data = data[batch_index, :, 2]
    
#     # Simulate the value for X0
#     simulated_X3 = np.zeros(k + 1)
#     simulated_X3[0] = X3_initial

#     for t in range(1, k + 1):
#         delta_X3 = compute_delta_X3(X0_data[t - 1],  X1_data[t - 1], X2_data[t - 1], simulated_X3[t - 1])
#         simulated_X3[t] = simulated_X3[t - 1] + 0.2 * delta_X3  # Perform update with step size 0.2

#     # Calculate residual squares for the current batch
#     residual_squares_all_batches[batch_index, :] = (simulated_X3 - data[batch_index, :, 3])**2

# # Calculate mean residual square error averaged by the number of batches for each time step
# mean_residual_squares = np.mean(residual_squares_all_batches, axis=0)

# # Plot the mean residual squares
# plt.figure(figsize=(10, 6))
# plt.plot(range(k + 1), mean_residual_squares , label='Mean Residual Squares', color='red', linestyle='--') # 
# # plt.plot(range(k + 1), data[batch_index, :, 2] , label='Squares', color='blue', linestyle='--') #
# plt.title('Mean Residual Squares for X3 Over Time Steps (Averaged Over Batches)')
# plt.xlabel('Time Step')
# plt.ylabel('Mean Residual Square')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Print mean residual square error across all time steps and batches
# overall_mean_residual_square = np.mean(mean_residual_squares)
# print(f"Overall Mean Residual Square Error for X3 over all time steps and batches: {overall_mean_residual_square}")