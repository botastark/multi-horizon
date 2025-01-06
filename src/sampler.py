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

# Sample the sensor model N times and calculate the average observation matrix
def sampler(true_matrix, altitude, N):
    rows, cols = true_matrix.shape
    cumulative_observation = np.zeros((rows, cols))

    for _ in range(N):
        observation_matrix = sensor_model(true_matrix, altitude)
        stat = calculate_statistics(true_matrix, observation_matrix)

        print(stat)
        # cumulative_observation += observation_matrix
        


    return cumulative_observation / N

# Calculate true positive, false negative, etc.
def calculate_statistics(true_matrix, averaged_observation):
    true_positive = np.sum((true_matrix == 1) & (averaged_observation >= 0.5))
    false_negative = np.sum((true_matrix == 1) & (averaged_observation < 0.5))
    false_positive = np.sum((true_matrix == 0) & (averaged_observation >= 0.5))
    true_negative = np.sum((true_matrix == 0) & (averaged_observation < 0.5))
    true = true_negative+true_positive
    all = true+ false_negative+false_positive
    return {
        'True Positive': true_positive,
        'False Negative': false_negative,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'true/all':true/all
    }
    

# Parameters
rows, cols = 10, 10  # Size of the grid
P_m = [0.50, 0.50]  # P(m=0) = 0.45, P(m=1) = 0.55


# Generate the true matrix
true_matrix = generate_true_matrix(rows, cols, P_m)
hs= [5, 10, 15, 20, 25, 30]
N_values = list(range(1,100,10))

plt.figure(figsize=(8, 6))
cmap = matplotlib.colormaps['viridis']
colors = [cmap(i / len(hs)) for i in range(len(hs))]

for i, altitude in enumerate(hs):
    # statistics_list = []
    acc= []
    for N in N_values:
        averaged_observation = sampler(true_matrix, altitude, N)
        stats = calculate_statistics(true_matrix, averaged_observation)
        # statistics_list.append(stats)
        acc.append(stats["true/all"])

    plt.plot(N_values, acc, label = f"h={altitude}", color= colors[i] )
    sig = sigma(altitude)
    plt.plot(N_values, [1-sig]*len(N_values), label = f"1-sigma h={altitude}", color= colors[i],linestyle='--')

plt.legend(title="Lines", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('acc of sensor model for limited budget')
plt.xlabel('budget: N samples')
plt.ylabel('accuracy: (TP+TN)/ALL')
plt.tight_layout()
plt.show()



# # Plot the results
# plt.figure(figsize=(8, 6))
# for i, N in enumerate(N_values):
#     averaged_observation = sampler(true_matrix, altitude, N)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(averaged_observation, cmap='viridis', interpolation='none')
#     plt.colorbar(label='Averaged Observation')
#     plt.title(f'N = {N}')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# Print statistics for each N

# for i, N in enumerate(N_values):
#     print(f"Statistics for N = {N}:")
    
#     for key, value in statistics_list[i].items():
#         print(f"  {key}: {value}")
#     print()
