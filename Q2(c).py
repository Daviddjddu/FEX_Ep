### FEX
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the MSE data
# file_path = 'seird_testing_data.npz'
# data = np.load(file_path)['data']

# # Simulation parameters
# dt = 0.2
# num_batches, num_time_steps, num_vars = data.shape
# k = num_time_steps - 1  # Number of time steps

# # The Nth predicted trajectory
# N = 5

# # Extract initial values
# X0_batches = data[:, 0, :]

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
#     X = np.zeros((steps + 1, num_vars))
#     X[0] = X0

#     for step in range(1, steps + 1):
#         delta_X = compute_delta_X(X[step - 1])
#         X[step] = X[step - 1] + dt * delta_X

#     return X

# # Perform the recursive update for each batch
# simulated_data = np.zeros((num_batches, k + 1, num_vars))
# for i in range(num_batches):
#     simulated_data[i] = recursive_update(X0_batches[i], k)    
    
# # Compute the MSE curves for all batches
# mse_curves = np.mean((simulated_data - data) ** 2, axis=2)  # Shape: (100, k+1)


# # Plot selected MSE curves
# plt.figure(figsize=(12, 8))

# # plt.plot(range(k + 1), mse_curves[N], color='Orange', linewidth=5, label = 'MSE', linestyle = "-.")

# # Plot selected prediction trajectories
# plt.plot(range(k + 1), simulated_data[N, :, 0], linewidth=5, label='Predicted S (Susceptible)', color='blue', linestyle = '--')
# plt.plot(range(k + 1), simulated_data[N, :, 1], linewidth=5, label='Predicted E (Exposed)', color='orange', linestyle = '-.')
# plt.plot(range(k + 1), simulated_data[N, :, 2], linewidth=5, label='Predicted I (Infectious)', color='red', linestyle = ':')
# plt.plot(range(k + 1), simulated_data[N, :, 3], linewidth=5, label='Predicted R (Recovered)', color='green')
# plt.plot(range(k + 1), simulated_data[N, :, 4], linewidth=5, label='Predicted D (Deceased)', color='black')


# # Labels and title
# plt.xlabel('Time Step', fontsize=28)
# plt.ylabel('Population Proportion', fontsize=28)
# # plt.yscale('log')
# plt.legend(loc='upper right', fontsize=20)
# plt.xticks(fontsize=28)  # Larger tick font
# plt.yticks(fontsize=28)  # Larger tick font
# plt.grid(True, alpha = 0.3)
# plt.savefig(f"Predicted_trajectory_FEX_SEIRD_{N+1}.png", dpi=300, bbox_inches="tight")

# # Display the plot
# plt.show()


### NN
