from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
import random

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using GPU
np.random.seed(seed)
random.seed(seed)

# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # LSTM layers with input size 4 (S, E, I, R) and hidden size 51
        self.lstm1 = nn.LSTMCell(4, 51)  # Input is 4 (S, E, I, R)
        self.lstm2 = nn.LSTMCell(51, 51)  # Hidden layer size 51
        self.linear = nn.Linear(51, 4)  # Output is 4 (S, E, I, R)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        # Process each time step in the input sequence
        for input_t in input.split(1, dim=1):
            input_t = input_t.squeeze(1)  # Remove the singleton dimension, [batch, 1, 4] -> [batch, 4]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        # Predict future time steps if specified
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50, help='steps to run')
    opt = parser.parse_args()

    # Load SEIR data (shaped [100, 250, 4]) using NumPy
    train_data = np.load('Data/seir_training_data.npz')['data']  # Load the data from the .npz file
    test_data = np.load('Data/seir_testing_data.npz')['data']  # Load the test data

    # Convert to PyTorch tensors and double precision
    train_data = torch.from_numpy(train_data).double()  # Training data to tensor
    test_data = torch.from_numpy(test_data).double()  # Testing data to tensor

    input = train_data[:, :-1, :]  # Training input: [100, 249, 4]
    target = train_data[:, 1:, :]  # Training target: [100, 249, 4]
    
    # Use the first time step from the test data correctly, shape: [100, 4]
    test_input = test_data[:, 0, :]  # Correctly extract the first time step as [100, 4]
    
    # True test data for comparison (excluding the first time step), shape: [100, 249, 4]
    test_true = test_data[:, 1:, :]  

    # Build the model
    seq = Sequence()
    total_params = sum(p.numel() for p in seq.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")
    seq.double()
    criterion = nn.MSELoss()

    # Use LBFGS as the optimizer
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    train_losses = []  # Store training losses per epoch
    
    # Begin training
    for i in range(opt.steps):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)  # Forward pass on the training data
            loss = criterion(out, target)  # Compute MSE loss
            loss.backward()
            return loss

        optimizer.step(closure)
        loss = optimizer.step(closure)  # The returned loss is the final loss for this step
        train_losses.append(loss.item())  # Store only the final loss of the step
        print('Training loss:', loss.item())

        # Begin to predict (use only the initial time step from the test data)
    with torch.no_grad():
        future = 248  # Number of future time steps to predict
        pred = seq(test_input.unsqueeze(1), future=future)  # Predict future based on initial value (add time dimension)
        y_pred = pred.detach().numpy()
        y_true = test_true.detach().numpy()

    # Reshape the initial input to match the format [batch_size, 1, 3]
    initial_input = test_input.unsqueeze(1).detach().numpy()  # Shape: [100, 1, 3]

    # Concatenate the initial time step to both the predicted and true sequences
    y_pred = np.concatenate((initial_input, y_pred), axis=1)  # Shape: [100, 250, 3]
    y_true = np.concatenate((initial_input, y_true), axis=1)  # Shape: [100, 250, 3]
    
    # Compute the MSE curves for all batches
    mse_curves = np.mean((y_true - y_pred) ** 2, axis=2)  # Shape: (100, k+1) 
    mean_mse_curve = np.mean(mse_curves, axis=0)
    std_mse_curve = np.std(mse_curves, axis=0)

    # Compute 95% confidence interval
    conf_interval = 1.96 * (std_mse_curve / np.sqrt(100))

    # Plot selected prediction
    plt.figure(figsize=(12, 8))

    # Plot selected prediction trajectories
    plt.plot(range(250), y_pred[5, :, 0], linewidth=5, label='Predicted S (Susceptible)', color='blue', linestyle = '--')
    plt.plot(range(250), y_pred[5, :, 1], linewidth=5, label='Predicted E (Exposed)', color='orange', linestyle = '-.')
    plt.plot(range(250), y_pred[5, :, 2], linewidth=5, label='Predicted I (Infectious)', color='red', linestyle = ':')
    plt.plot(range(250), y_pred[5, :, 3], linewidth=5, label='Predicted R (Recovered)', color='green')

    # Labels and title
    plt.xlabel('Time Step', fontsize=28)
    plt.ylabel('Population Proportion', fontsize=28)
    # plt.yscale('log')
    plt.legend(loc='upper right', fontsize=20)
    plt.xticks(fontsize=28)  # Larger tick font
    plt.yticks(fontsize=28)  # Larger tick font
    plt.grid(True, alpha = 0.3)
    plt.savefig(f"Plot/Predicted_trajectory_RNN_SEIR_{6}.png", dpi=300, bbox_inches="tight")

    mse_per_time_step = []
    for t in range(250):
        mse = mean_squared_error(y_true[:, t, :], y_pred[:, t, :])
        mse_per_time_step.append(mse)
    
    # Save training losses for future plotting
    np.save('Data/seir_train_losses_RNN.npy', np.array(train_losses))
    np.save('Data/mse_RNN_SEIR.npy', mean_mse_curve)
    np.save("Data/modified_mse_RNN_SEIR.npy", conf_interval)

    # # Save MSEs for each time step
    # np.save('mse_RNN_SEIR.npy', mse_per_time_step)
    
#     # Draw the results for the first batch
#     plt.figure(figsize=(30, 10))
#     plt.title('Predict future values for SEIR model\n(Dashlines are predicted values)', fontsize=30)
#     plt.xlabel('Time Steps', fontsize=20)
#     plt.ylabel('Values', fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)

#     def draw(true, pred, color, label):
#         plt.plot(np.arange(true.shape[0]), true, color, label=f'True {label}', linewidth=2.0)
#         plt.plot(np.arange(future+2), pred, color + ':', label=f'Predicted {label}', linewidth=2.0)

#     # Draw results for the first test example
#     draw(y_true[0, :, 0], y_pred[0, :, 0], 'r', 'S')
#     draw(y_true[0, :, 1], y_pred[0, :, 1], 'g', 'E')
#     draw(y_true[0, :, 2], y_pred[0, :, 2], 'b', 'I')
#     draw(y_true[0, :, 3], y_pred[0, :, 3], 'y', 'R')

#     plt.legend()
#     plt.savefig('seir_predict_%d.pdf' % 1)
#     plt.close()

    # # Plot the MSE over time steps
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(50), mse_per_time_step, label='MSE', color='blue', linewidth=2.0)
    # plt.title('Mean Squared Error (MSE) over Time Steps', fontsize=18)
    # plt.xlabel('Time Steps', fontsize=14)
    # plt.ylabel('MSE', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
