import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# Load the data from the CSV file
file_path = 'Data/Hubei_fex_normal_full.csv'
data = pd.read_csv(file_path)

# Extract the nth row (index n-1) as a NumPy array for the initial values
# X0 = data.iloc[85].to_numpy()
n = 0
X0 = data.iloc[n].to_numpy() # Used to show training accuracy
def compute_delta_X(X_current):
    X0 = X_current[0]
    X1 = X_current[1]
    X2 = X_current[2]

    # Calculate delta_X0 (done)
    delta_X0 = (
        # (((((0.3677*((X0)**2)+-1.3812*((X1)**2)+0.5442*((X2)**2)+0.0322))-((0.1882*((X0)**4)+-2.1678*((X1)**4)+0.3310*((X2)**4)+0.0238))))*((-1.7933*((X0)**3)+4.8552*((X1)**3)+-0.0613*((X2)**3)+0.0448)))
        (((((-0.9030*((X0)**3)+2.4025*((X1)**3)+-0.0262*((X2)**3)+0.0311))*((-0.1840*((X0)**3)+-0.0432*((X1)**3)+-2.5147*((X2)**3)+-0.0181))))*((0.1919*(np.sin(X0))+0.1812*(np.sin(X1))+0.7006*(np.sin(X2))+-0.7283)))
    )
    delta_X1 = (
        # (((((0.3561*(X0)+-0.7692*(X1)+0.2030*(X2)+0.1901))*((2.3852*(np.sin(X0))+-2.7112*(np.sin(X1))+0.0196*(np.sin(X2))+1.4565))))*((-1.3774*(np.sin(X0))+2.3038*(np.sin(X1))+-0.5204*(np.sin(X2))+0.0021)))
        (((((-1.5383*(np.exp(X0))+1.3790*(np.exp(X1))+-0.0797*(np.exp(X2))+-0.9186))*((-0.4292*(X0)+0.9682*(X1)+-0.2423*(X2)+-0.2602))))*((-0.8972*(X0)+1.4067*(X1)+-0.2637*(X2)+0.0054)))
    )
    delta_X2 = (
        # ((((-0.6660 *(np.cos(X0))-0.0227*(np.cos(X1))+0.7341*(np.cos(X2))-0.0205))+((1.5332*(X0)-2.6224*(X1)+0.8284*(X2)-0.026))))*((1.3249*(X0)-0.0831*(X1))+0.9974*(X2)-1.1279)
        # (((((1.8152*(np.sin(X0))+-2.4195*(np.sin(X1))+0.6940*(np.sin(X2))+-0.0010))*((-2.1982*((X0)**2)+-0.2551*((X1)**2)+-0.6258*((X2)**2)+0.6900))))*((0.8736*((X0)**3)+2.0320*((X1)**3)+-3.4589*((X2)**3)+-1.6054)))
        (((((1.6211*(np.sin(X0))+-2.1673*(np.sin(X1))+0.6259*(np.sin(X2))+-0.0008))*((4.3940*((X0)**3)+1.6576*((X1)**3)+0.5903*((X2)**3)+-0.6928))))*((0.1196*((X0)**2)+-3.7194*((X1)**2)+1.7070*((X2)**2)+1.7746)))
    )                                             
    return np.array([delta_X0, delta_X1, delta_X2])

def recursive_update(X0, steps):
    X = np.zeros((steps, 3))  # Initialize an array of zeros for 18 steps (from row 82)
    X[0] = X0  # Start with the 82rd row's values

    for step in range(1, steps):
        delta_X = compute_delta_X(X[step - 1])
        X[step] = X[step - 1] + delta_X
        X[step][X[step] < 0] = 0  # Prevent values from going below zero
    return X

# Define the total number of rows as 100, and steps as 15 (100 - 85)
total_rows = 100
steps = total_rows - n  # 15 steps from row 85 to 100

# Perform the recursive update using the initial values from row 8
X_values_n_onwards = recursive_update(X0, steps)

# Initialize X_values with zeros for the first 82 rows and then append the computed values
X_values = np.zeros((total_rows, 3))  # 100 rows in total
X_values[n:] = X_values_n_onwards  # Fill from row n+1 onward with computed values

# Scaling
scaling_factors = np.array([64435, 4512, 50633])
X_values = X_values * scaling_factors

# Print the dimensions of X_values
print("Dimensions of X_values:", X_values.shape)

# Ensure the dataset is time-aligned with the simulated values, convert to NumPy array
actual_values = data[['Recovered', 'Deaths', 'Active']].to_numpy()
actual_values = actual_values * scaling_factors

# Calculate mean square error (MSE) starting from the 82rd row onward
mse = np.mean((X_values[n:] - actual_values[n:]) ** 2, axis=0)

# Print mean square error for each quantity
quantities = ['Recovered', 'Deaths', 'Active']
for i, quantity in enumerate(quantities):
    print(f"Mean square error for {quantity} (starting from row 85): {mse[i]:.2f}")

# Function to format the y-axis scale
def scientific_formatter(x, pos):
    return f'{x/1e4:.0f}'

# Load simulated result using SEIQRDP method
mat1 = scipy.io.loadmat('Data/Q1.mat') 
mat2 = scipy.io.loadmat('Data/R1.mat')
mat3 = scipy.io.loadmat('Data/D1.mat')

data1 = mat1['Q'].T  
data2 = mat2['R'].T
data3 = mat3['D'].T

# Starting from the first term
start_index = 0

# Plotting the evolution of X over time
plt.figure(figsize=(12, 8))

# Plot the actual data from the CSV file with triangles and crosses
plt.plot(
    range(start_index, total_rows), 
    actual_values[start_index:, 0], 
    label='Actual Recover', 
    color='blue', 
    marker='^', 
    markerfacecolor='none', 
    linestyle='none',
    linewidth=5
)
plt.plot(
    range(start_index, total_rows), 
    actual_values[start_index:, 1], 
    label='Actual Decease', 
    color='black', 
    marker='x', 
    markerfacecolor='none', 
    linestyle='none',
    linewidth=5
)
plt.plot(
    range(start_index, total_rows), 
    actual_values[start_index:, 2], 
    label='Actual Active', 
    color='red', 
    marker='o', 
    markerfacecolor='none', 
    linestyle='none',
    linewidth=5
)

# Plot the predicted data
plt.plot(
    range(start_index, 85), 
    X_values[start_index:85, 0], 
    label='Recover', 
    color='blue',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(start_index, 85), 
    X_values[start_index:85, 1], 
    label='Decease', 
    color='black',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(start_index, 85), 
    X_values[start_index:85, 2], 
    label='Active',
    color='red',
    linestyle='--',
    linewidth=5
)

plt.plot(
    range(85, len(X_values)), 
    X_values[85:, 0], 
    label='Predict_R', 
    color='blue', 
    linestyle='-.',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)), 
    X_values[85:, 1], 
    label='Predict_D', 
    color='black', 
    linestyle='-.',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)), 
    X_values[85:, 2], 
    label='Predict_A', 
    color='red', 
    linestyle='-.',
    linewidth=5
)

# Update x-axis ticks and labels to days
day_ticks = [0, 24, 49, 74, 99]  # Positions corresponding to the labels
day_labels = ['Day0', 'Day24', 'Day49', 'Day74', 'Day99']  # Custom labels
plt.xticks(ticks=day_ticks, labels=day_labels, fontsize=20)

# Set x-axis limits to remove space before "Feb" and after "May"
plt.xlim(0, 100)

# Set scientific notation formatter for the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

# Update labels and title
plt.xlabel('Time (days)', fontsize=20)
plt.ylabel('Number of cases (×10⁴)', fontsize=20)
plt.legend()
plt.xticks(fontsize=16)  # Larger tick font
plt.yticks(fontsize=16)  # Larger tick font
plt.grid(True, alpha=0.3)


# Save and show the plot
plt.savefig('Plot/Covid_prediction_scaled_day_labels_100.png')

# # Get the last element of the vector
# last_element1 = data1[-1, 0]

# # Extend data3 to match the length of X_values
# extended_data1 = np.full((len(X_values)-start_index, 1), last_element1)
# extended_data1[:len(data1), 0] = data1[:, 0]

# # Get the last element of the vector
# last_element2 = data2[-1, 0]

# # Extend data3 to match the length of X_values
# extended_data2 = np.full((len(X_values)-start_index, 1), last_element2)
# extended_data2[:len(data2), 0] = data2[:, 0]

# # Get the last element of the vector
# last_element3 = data3[-1, 0]

# # Extend data3 to match the length of X_values
# extended_data3 = np.full((len(X_values)-start_index, 1), last_element3)
# extended_data3[:len(data3), 0] = data3[:, 0]

# Plot 1: Blue (Recovered)
plt.figure(figsize=(12, 8))
plt.plot(
    range(start_index, total_rows),
    actual_values[start_index:, 0],
    label='Actual Recover',
    color='purple',
    marker='^',
    markerfacecolor='none',
    markersize=10,
    linestyle='none',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    X_values[start_index:85, 0],
    label='FEX_train',
    color='blue',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    X_values[85:, 0],
    label='FEX_predict',
    color='red',
    linestyle='-.',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    data2[:85],
    label='SEIQRDP_train',
    color='orange',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    data2[85:],
    label='SEIQRDP_predict',
    color='green',
    linestyle='-.',
    linewidth=5
)

plt.xticks(ticks=day_ticks, labels=day_labels, fontsize = 28)
plt.yticks(fontsize=28)
plt.xlim(0, 100)
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
plt.xlabel('Time (days)', fontsize=28)
plt.ylabel('Number of cases (×10⁴)', fontsize=28)
plt.legend(loc='lower right', fontsize=28)
plt.grid(True, alpha = 0.3)
plt.savefig('Plot/Covid_Recovered.png', dpi=500, bbox_inches="tight")
plt.close()

# Plot 2: Black (Deceased)
plt.figure(figsize=(12, 8))
plt.plot(
    range(start_index, total_rows),
    actual_values[start_index:, 1],
    label='Actual Decease',
    color='purple',
    marker='x',
    markerfacecolor='none',
    markersize=10,
    linestyle='none',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    X_values[start_index:85, 1],
    label='FEX_train',
    color='blue',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    X_values[85:, 1],
    label='FEX_predict',
    color='red',
    linestyle='-.',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    data3[:85],
    label='SEIQRDP_train',
    color='orange',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    data3[85:],
    label='SEIQRDP_predict',
    color='green',
    linestyle='-.',
    linewidth=5
)

plt.xticks(ticks=day_ticks, labels=day_labels, fontsize = 28)
plt.yticks(fontsize=28)
plt.xlim(0, 100)
# plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
plt.xlabel('Time (days)', fontsize = 28)
plt.ylabel('Number of cases', fontsize = 28)
plt.legend(fontsize = 28)
plt.grid(True, alpha = 0.3)
plt.savefig('Plot/Covid_Deceased.png', dpi=500, bbox_inches="tight")
plt.close()  

# Plot 3: Red (Active)
plt.figure(figsize=(12,8))
plt.plot(
    range(start_index, total_rows),
    actual_values[start_index:, 2],
    label='Actual Active',
    color='purple',
    marker='o',
    markerfacecolor='none',
    markersize=10,
    linestyle='none',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    X_values[start_index:85, 2],
    label='FEX_train',
    color='blue',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    X_values[85:, 2],
    label='FEX_predict',
    color='red',
    linestyle='-.',
    linewidth=5
)
plt.plot(
    range(start_index, 85),
    data1[:85],
    label='SEIQRDP_train',
    color='orange',
    linestyle='--',
    linewidth=5
)
plt.plot(
    range(85, len(X_values)),
    data1[85:],
    label='SEIQRDP_predict',
    color='green',
    linestyle='-.',
    linewidth=5
)

plt.xticks(ticks=day_ticks, labels=day_labels, fontsize = 28)
plt.yticks(fontsize=28)
plt.xlim(0, 100)
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
plt.xlabel('Time (days)', fontsize = 28)
plt.ylabel('Number of cases (×10⁴)', fontsize = 28)
plt.legend(loc='right', fontsize = 28)
plt.grid(True, alpha = 0.3)
plt.savefig('Plot/Covid_Active.png', dpi=500, bbox_inches="tight")
plt.close()