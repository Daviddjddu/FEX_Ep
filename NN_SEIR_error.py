import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import random

# Set seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using GPU
np.random.seed(seed)
random.seed(seed)
# Function to prepare input-output pairs for regression (for S, E, I, R dimensions)
def prepare_data(trajectories, dt):
    num_simulations, num_steps, num_features = trajectories.shape
    X = []
    y = []

    for i in range(num_simulations):
        for j in range(num_steps - 1):
            # Append the current state (S_i, E_i, I_i, R_i) to X
            X.append(trajectories[i, j, :4])  # Taking first four columns (S, E, I, R)
            # Calculate the rate of change for four dimensions and append to y
            y.append((trajectories[i, j+1, :4] - trajectories[i, j, :4]) / dt)

    X = np.array(X)
    y = np.array(y)

    return X, y

# Load the data from the two .npz files
train_data = np.load('Data/seir_training_data.npz')['data']  # Updated dataset file for SEIR model
test_data = np.load('Data/seir_testing_data.npz')['data']    # Updated dataset file for SEIR model

# Define the time step as dt
dt = 0.2  

# Prepare data for training and testing (for S, E, I, R dimensions)
X_train, y_train = prepare_data(train_data, dt)
X_test, y_test = prepare_data(test_data, dt)

# Convert data to PyTorch tensors (double precision)
X_train = torch.tensor(X_train, dtype=torch.float64)
y_train = torch.tensor(y_train, dtype=torch.float64)
X_test = torch.tensor(X_test, dtype=torch.float64)
y_test = torch.tensor(y_test, dtype=torch.float64)

class SEIRNet(nn.Module):
    def __init__(self):
        super(SEIRNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input layer: 4 features (S, E, I, R)
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, 4)  # Output layer: 4 features (dS/dt, dE/dt, dI/dt, dR/dt)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the model and set it to double precision
model = SEIRNet().double()  # This will set all layers to float64
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
batch_size = 32

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1

    avg_epoch_loss = epoch_loss / batch_count
    train_losses.append(avg_epoch_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

torch.save(model.state_dict(), 'Data/seir_regression_model.pt')

# Test the model using the initial state and predict future values iteratively
model.eval()

def simulate_future_values(model, X_initial, num_steps, dt):
    """Simulates future values using the model and initial values."""
    X_simulated = [X_initial]

    current_state = X_initial.clone().double()  # Ensure it's in double precision
    for _ in range(num_steps):
        rate_of_change = model(current_state)  # Predict dS/dt, dE/dt, dI/dt, dR/dt
        next_state = current_state + dt * rate_of_change  # Update state
        X_simulated.append(next_state)
        current_state = next_state
    
    return torch.stack(X_simulated, dim=1)  # Stack over time

# Convert X_initial to double precision
X_initial = torch.tensor(test_data[:, 0, :4], dtype=torch.float64)

# Simulate future values for the entire test set
num_time_steps = test_data.shape[1] - 1
X_simulated = simulate_future_values(model, X_initial, num_time_steps, dt)

X_simulated = X_simulated.detach().numpy()
# Plot selected MSE curves
plt.figure(figsize=(12, 8))

# Plot selected prediction trajectories
plt.plot(range(250), X_simulated[5, :, 0], linewidth=5, label='Predicted S (Susceptible)', color='blue', linestyle = '--')
plt.plot(range(250), X_simulated[5, :, 1], linewidth=5, label='Predicted E (Exposed)', color='orange', linestyle = '-.')
plt.plot(range(250), X_simulated[5, :, 2], linewidth=5, label='Predicted I (Infectious)', color='red', linestyle = ':')
plt.plot(range(250), X_simulated[5, :, 3], linewidth=5, label='Predicted R (Recovered)', color='green')

# Labels and title
plt.xlabel('Time Step', fontsize=28)
plt.ylabel('Population Proportion', fontsize=28)
# plt.yscale('log')
plt.legend(loc='upper right', fontsize=20)
plt.xticks(fontsize=28)  # Larger tick font
plt.yticks(fontsize=28)  # Larger tick font
plt.grid(True, alpha = 0.3)
plt.savefig(f"Plot/Predicted_trajectory_NN_SEIR_{6}.png", dpi=300, bbox_inches="tight")

# Calculate the MSE for each time step (for SEIR model, 4 dimensions)
y_true = test_data[:, 0:, :4]  # True values starting from time step 0

# Compute the MSE curves for all batches
mse_curves = np.mean((y_true - X_simulated) ** 2, axis=2)  # Shape: (100, k+1) 
mean_mse_curve = np.mean(mse_curves, axis=0)
std_mse_curve = np.std(mse_curves, axis=0)

# Compute 95% confidence interval
conf_interval = 1.96 * (std_mse_curve / np.sqrt(100))

np.save('Data/mse_NN_SEIR.npy', mean_mse_curve)
np.save("Data/modified_mse_NN_SEIR.npy", conf_interval)
np.save('Data/seir_train_losses_NN.npy', np.array(train_losses))