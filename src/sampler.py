import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# Define the sigma function
def sigma(altitude):
    a = 1
    b = 0.05
    return a * (1 - np.exp(-b * altitude))

# Generate a true matrix based on P_m
def generate_true_matrix(rows, cols, P_m):
    return np.random.choice([0, 1], size=(rows, cols), p=P_m)

# Simulate the sensor model to create an observation matrix z
def sensor_model(true_matrix, altitude):
    sig = sigma(altitude)
    P_z_equals_m = 1 - sig
    P_z_not_equals_m = sig

    rows, cols = true_matrix.shape
    observation_matrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if true_matrix[i, j] == 1:
                observation_matrix[i, j] = np.random.choice([1, 0], p=[P_z_equals_m, P_z_not_equals_m])
            else:
                observation_matrix[i, j] = np.random.choice([0, 1], p=[P_z_equals_m, P_z_not_equals_m])

    return observation_matrix

# Calculate true positive, false negative, etc.
def calculate_statistics(true_matrix, averaged_observation):
    true_positive = np.sum((true_matrix == 1) & (averaged_observation >= 0.5))
    false_negative = np.sum((true_matrix == 1) & (averaged_observation < 0.5))
    false_positive = np.sum((true_matrix == 0) & (averaged_observation >= 0.5))
    true_negative = np.sum((true_matrix == 0) & (averaged_observation < 0.5))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracy = (true_positive + true_negative) / true_matrix.size

    return {
        'True Positive': true_positive,
        'False Negative': false_negative,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'F1 Score': f1_score,
        'Accuracy': accuracy}

# Sample the sensor model N times and calculate the average observation matrix
def sampler_avg_obs(true_matrix, altitude, N):
    rows, cols = true_matrix.shape
    cumulative_observation = np.zeros((rows, cols))

    for _ in range(N):
        observation_matrix = sensor_model(true_matrix, altitude)
        cumulative_observation += observation_matrix
    return cumulative_observation / N

# Sample the sensor model N times and calculate the average 1-sigma
def sampler_avg_sigma(true_matrix, altitude, N):
    inv_sigma = 0
    for _ in range(N):
        observation_matrix = sensor_model(true_matrix, altitude)
        stat = calculate_statistics(true_matrix, observation_matrix)
        inv_sigma += stat['Accuracy']

    return  inv_sigma / N

# Convergence Detection
def find_convergence_N(running_averages, tolerance=0.01, window_size=10):
    for i in range(window_size, len(running_averages)):
        window_std = np.std(running_averages[i - window_size:i])
        if window_std < tolerance:
            return i
    return len(running_averages)


def plot_truth_and_observations(true_matrix, averaged_observation, h, threshold):
    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Visualization h={h}', fontsize=16)
    
    binarized_observation = (averaged_observation >= threshold).astype(int)
    
    # Ground truth
    im1 = axes[0].imshow(true_matrix, cmap='Greys', interpolation='nearest')
    axes[0].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Average observation
    im2 = axes[1].imshow(averaged_observation, cmap='coolwarm', interpolation='nearest')
    axes[1].set_title('Average Observation')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Confidence
    im3 = axes[2].imshow(true_matrix-binarized_observation, cmap='viridis', interpolation='nearest')
    axes[2].set_title('diff true-bin Map')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Set common properties
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def plot_acc(hs, N_values, accuracy_data_sigma, convergence_N_values, accuracy_data_obs):
    max_N = max(convergence_N_values.values())+2
    cmap = matplotlib.colormaps['viridis']
    colors = [cmap(i / len(hs)) for i in range(len(hs))]
    # max_N = len(N_values)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    # Plot the final accuracy graph after collecting all data
    for i, altitude in enumerate(hs):
        acc_obs = accuracy_data_obs[altitude]
        acc_s = accuracy_data_sigma[altitude]
        if acc_obs != []:
            ax1.plot(N_values[0:len(acc_obs)], acc_obs, label=f"h={altitude} obs", color=colors[i], linestyle='-.')
        ax1.plot(N_values[0:len(acc_s)], acc_s, label=f"h={altitude} avg sigma", color=colors[i], linestyle='-')
        sig = sigma(altitude)
        ax1.plot(N_values[0:max_N], [1 - sig] * max_N, label=f"1-sigma h={altitude}", color=colors[i], linestyle='--')
        conv_N = convergence_N_values[altitude]
        ax1.axvline(x=convergence_N_values[altitude], color=colors[i], linestyle=':', alpha=0.7, label=f"Converged h={altitude}")
        ax1.text(
                conv_N + 0.5, 1 - sig,  # Position slightly right of the line and near 1-sigma
                f"N={conv_N} for h={altitude}", color="black", fontsize=10
            )
    ax1.legend(title="Lines", bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title(f'acc of sensor model for limited budget (window size {window_size} and std thr{std_th})')
    ax1.set_xlabel('budget: N samples')
    ax1.set_ylabel('accuracy')
    fig1.tight_layout()
    plt.show()

def true_map_size(altitude):
    fov = np.pi / 3
    x_angle = fov / 2  # rads
    y_angle = fov / 2  # rads
    grid_length = 0.125
    rows = altitude * math.tan(x_angle)
    cols = altitude * math.tan(y_angle)
    rows = int(round(rows / grid_length)  )
    cols = int(round(cols / grid_length) )
    return (rows, cols)

# Parameters
rows, cols = 1,1  # Size of the grid
P_m = [0.50, 0.50]  # P(m=0) = 0.45, P(m=1) = 0.55
hs = [ 5.41, 10.83, 16.24, 21.65, 27.06, 32.48]
N_values = list(range(1,101,1))
window_size = 3
std_th = 0.001



accuracy_data_obs = {}
accuracy_data_sigma = {}
convergence_N_values = {}
for i, altitude in enumerate(hs):
    # rows, cols = true_map_size(altitude)
    if rows==1 and cols==1:
        true_matrix = np.array([0,1])
        true_matrix = np.expand_dims(true_matrix, axis=0)
        N_values = N_values[0:len(N_values)]
    else:
        true_matrix = generate_true_matrix(rows, cols, P_m)
    acc_avg_obs = []
    acc_avg_sigma= []
    running_averages = []
    for N in N_values:
        averaged_observation = sampler_avg_obs(true_matrix, altitude, N)
        stats = calculate_statistics(true_matrix, averaged_observation)
        acc_avg_obs.append(stats["Accuracy"])
        acc_avg_sigma.append(sampler_avg_sigma(true_matrix, altitude, N))
        # Compute running average
        if altitude not in convergence_N_values:
            running_averages.append(np.mean(acc_avg_sigma))
            if len(running_averages) > window_size:  # Check condition after building the window
                window_std = np.std(running_averages[-window_size:])  # Calculate std in the last 10
                if window_std < std_th :  # Convergence detected
                    convergence_N_values[altitude] = N
                    break  # Exit the loop when convergence is found
    else:
        convergence_N_values[altitude] = N_values[-1]

    # plot_truth_and_observations(true_matrix, averaged_observation, altitude, 0.5)
    accuracy_data_obs[altitude] = acc_avg_obs
    accuracy_data_sigma[altitude] = acc_avg_sigma

print(convergence_N_values)
plot_acc(hs, N_values, accuracy_data_sigma, convergence_N_values, accuracy_data_obs)