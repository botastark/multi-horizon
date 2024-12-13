import re
import matplotlib.pyplot as plt


def extract_values_from_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()

    pattern = r"(\d+)\s+([\d\.]+)\s+([\d\.]+)"

    steps = re.findall(pattern, text)

    entropy_values = [float(step[1]) for step in steps]
    mse_values = [float(step[2]) for step in steps]
    return entropy_values, mse_values


def plot_entropy_mse_combined(entropy1, entropy2, mse1, mse2, first_plot = "MexGen adaptive", second_plot="IG adaptive"):
    steps1 = list(range(len(entropy1)))
    steps2 = list(range(len(entropy2)))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(
        steps1,
        entropy1,
        label="Entropy " + first_plot,
        color="blue",
    )
    ax[0].plot(
        steps2,
        entropy2,
        label="Entropy " + second_plot,
        color="green",
    )
    ax[0].set_title("Entropy vs. Step")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Entropy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(steps1, mse1, label="MSE "+first_plot, color="red")
    ax[1].plot(steps2, mse2, label="MSE "+second_plot, color="purple")
    ax[1].set_title("MSE vs. Step")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("MSE")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()


file_path = "/home/bota/Desktop/active_sensing/equal_mexgen/ig_with_mexgen_equal_1.txt"
entropy_mexgen, mse_mexgen = extract_values_from_file(file_path)
file_path_ = "/home/bota/Desktop/active_sensing/equal/ig_equal_1.txt"
entropy_, mse_ = extract_values_from_file(file_path_)

plot_entropy_mse_combined(entropy_mexgen, entropy_, mse_mexgen, mse_)
