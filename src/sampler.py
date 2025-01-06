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
    avg_acc = 0

    for _ in range(N):
        observation_matrix = sensor_model(true_matrix, altitude)
        stat = calculate_statistics(true_matrix, observation_matrix)
        avg_acc+=stat['true/all']
        cumulative_observation += observation_matrix
    return cumulative_observation / N, avg_acc/N

# Calculate true positive, false negative, etc.
def calculate_statistics(true_matrix, averaged_observation, Pm=None):
    diff = 0.5
    if Pm is not None:
        diff = Pm[0]

    true_positive = np.sum((true_matrix == 1) & (averaged_observation >= diff))
    false_negative = np.sum((true_matrix == 1) & (averaged_observation < diff))
    false_positive = np.sum((true_matrix == 0) & (averaged_observation >=diff))
    true_negative = np.sum((true_matrix == 0) & (averaged_observation < diff))
    true = true_negative + true_positive
    all = true + false_negative + false_positive
    return {
        'True Positive': true_positive,
        'False Negative': false_negative,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'true/all':true/all
    }
    

# Parameters
rows, cols = 1, 1  # Size of the grid
P_m = [0.50, 0.50]  # P(m=0) = 0.45, P(m=1) = 0.55
P_m_decide = [0.450, 0.550]  # P(m=0) = 0.45, P(m=1) = 0.55


# Generate the true matrix
true_matrix = generate_true_matrix(rows, cols, P_m)
hs= [5, 10, 15, 20, 25, 30]
N_values = list(range(1,200,1))
# N_values = [1,10,25,50,100,150, 200]

plt.figure(figsize=(8, 6))
cmap = matplotlib.colormaps['viridis']
colors = [cmap(i / len(hs)) for i in range(len(hs))]

for i, altitude in enumerate(hs):
    acc= []
    acc_avg = []

    for N in N_values:
        averaged_observation, avg_acc = sampler(true_matrix, altitude, N)
        stats = calculate_statistics(true_matrix, averaged_observation)
        # acc.append(stats["true/all"])
        if true_matrix[0][0]==1:
            acc_avg.append(averaged_observation[0])
        else:
            acc_avg.append(1-averaged_observation[0])

    # plt.plot(N_values, acc, label = f"h={altitude}", color= colors[i] )
    plt.plot(N_values, acc_avg, label = f"h={altitude}", color= colors[i] )
    sig = sigma(altitude)
    plt.plot(N_values, [1-sig]*len(N_values), label = f"h={altitude} 1-sigma={round(1-sig,2)}", color= colors[i],linestyle='--')
    print(f"h={altitude}, avgobs = {averaged_observation[0]}, 1-sig={1-sig}")
plt.legend(title="Lines", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('acc of sensor model for limited budget')
plt.xlabel('budget: N samples')
# plt.xscale("log")

plt.ylabel('accuracy: (TP+TN)/ALL')
plt.tight_layout()
plt.show()

# Print statistics for each N

# for i, N in enumerate(N_values):
#     print(f"Statistics for N = {N}:")
    
#     for key, value in statistics_list[i].items():
#         print(f"  {key}: {value}")