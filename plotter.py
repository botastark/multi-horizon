import glob
import os
import re
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import glob

import matplotlib.pyplot as plt
import pandas as pd
import glob


dir = "/home/bota/Desktop/active_sensing/results"
dir = "/home/bota/Desktop/active_sensing/cache/tester_cache"


def parse_file_to_table(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse Pairwise and Gaussian radius
    pairwise = None
    gaussian_radius = None
    error_margin = None
    for line in lines:
        if line.startswith("Pairwise:"):
            pairwise = line.split(":")[1].strip()
        elif line.startswith("Error margin:"):
            error_margin = line.split(":")[1].strip()
            if error_margin == "None":
                error_margin = 0.0
            else:
                error_margin = float(error_margin)
        elif "Gaussian radius" in line:
            parts = line.split()
            gaussian_radius = int(parts[-1])  # The last element is the radius

    if pairwise is None or gaussian_radius is None:
        raise ValueError(
            f"Pairwise  {pairwise}, Error Margin {error_margin} or Gaussian radius {gaussian_radius} not found in the file: {file_path}"
        )

    # Locate the starting point of the table
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step"):
            start_index = i + 2
            break

    if start_index is None:
        raise ValueError(f"Table header 'Step' not found in the file: {file_path}")

    table_data = []
    for line in lines[start_index:]:
        if not line.strip() or not line[0].isdigit():
            continue
        parts = line.split()
        step, entropy, mse, height, coverage = map(float, parts[:5])
        table_data.append([int(step), entropy, mse, height, coverage])

    df = pd.DataFrame(
        table_data, columns=["Step", "Entropy", "MSE", "Height", "Coverage"]
    )
    df["Pairwise"] = pairwise
    df["ErrorMargin"] = error_margin
    df["GaussianRadius"] = gaussian_radius
    return df


def aggregate_data_by_settings(folder_path):
    file_paths = glob.glob(f"{folder_path}/*.txt")
    # error_margin = extract_error_margin_from_folder(folder_path)
    all_data = [parse_file_to_table(file_path) for file_path in file_paths]
    # all_data = []
    # for file_path in file_paths:
    #     data = parse_file_to_table(file_path)  # Assuming this is a function you have
    #     data["ErrorMargin"] = error_margin  # Add the error margin to the data
    #     all_data.append(data)

    combined_df = pd.concat(all_data)
    return (
        combined_df.groupby(["Pairwise", "GaussianRadius", "ErrorMargin", "Step"])
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_category_stats_by_settings(stats, categories):

    radii = stats["GaussianRadius"].unique()
    pairwises = stats["Pairwise"].unique()
    error_margins = stats["ErrorMargin"].unique()
    pairs = list(product(radii, pairwises, error_margins))
    unique_settings = set(pairs)

    for radius, pairwise, error_margin in unique_settings:
        setting_data = stats[
            (stats["Pairwise"] == pairwise)
            & (stats["GaussianRadius"] == radius)
            & (stats["ErrorMargin"] == error_margin)
        ]

        steps = setting_data["Step"]
        fig, ax = plt.subplots(len(categories), 1, figsize=(7, 3 * len(categories)))

        if len(categories) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one category

        for i, category in enumerate(categories):
            mean_values = setting_data[(category, "mean")]
            std_values = setting_data[(category, "std")]

            ax[i].plot(steps, mean_values, label=f"{category} Mean", color="blue")
            ax[i].fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color="blue",
                alpha=0.2,
                label=f"{category} Std Dev",
            )

            ax[i].set_title(
                f"{category} for Pairwise={pairwise}, Radius={radius} and Error Margin={error_margin}"
            )
            ax[i].set_xlabel("Steps")
            ax[i].set_ylabel(category)
            ax[i].legend()
            ax[i].grid()

        plt.tight_layout()
        plt.show()


def plot_category_stats_by_e(stats, unique_settings):

    categories = ["Entropy", "MSE", "Height", "Coverage"]
    radius, pairwise = unique_settings
    error_margins = [0.0, 0.05, 0.1, 0.3]
    # radii = stats["GaussianRadius"].unique()
    # pairwises = stats["Pairwise"].unique()
    # error_margins = stats["ErrorMargin"].unique()
    # pairs = list(product(radii, pairwises, error_margins))
    # unique_settings = set(pairs)

    # for radius, pairwise, error_margin in unique_settings:
    fig, ax = plt.subplots(len(categories), 1, figsize=(7, 3 * len(categories)))
    colors = ["blue", "red", "green", "purple"]

    for iter, error_margin in enumerate(error_margins):

        setting_data = stats[
            (stats["Pairwise"] == pairwise)
            & (stats["GaussianRadius"] == radius)
            & (stats["ErrorMargin"] == error_margin)
        ]
        if error_margin == None:
            print(setting_data)

        steps = setting_data["Step"]

        if len(categories) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one category

        for i, category in enumerate(categories):
            mean_values = setting_data[(category, "mean")]
            std_values = setting_data[(category, "std")]

            ax[i].plot(
                steps, mean_values, label=f"E={error_margin}", color=colors[iter]
            )
            ax[i].fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color=colors[iter],
                alpha=0.1,
                # label=f"{category} Std Dev",
            )

            ax[i].set_title(
                f"{category} for Pairwise={pairwise}, Radius={radius} and Error Margins={error_margins}"
            )
            ax[i].set_xlabel("Steps")
            ax[i].set_ylabel(category)
            ax[i].legend()
            ax[i].grid()

    plt.tight_layout()
    plt.show()


# folders = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
folders = dir
all_stats = pd.DataFrame()

stats = aggregate_data_by_settings(dir)
all_stats = pd.concat([all_stats, stats], ignore_index=True)  # Append to all_stats
# all_stats
print(all_stats.head())
print(f"unique error margins: {all_stats['ErrorMargin'].unique()}")
print(f"unique rad: {all_stats['GaussianRadius'].unique()}")
print(f"unique pairwise: {all_stats['Pairwise'].unique()}")
# all_stats
# setting_data = all_stats[(stats["ErrorMargin"] == 0.0)]
# print(setting_data)
# plot_category_stats_by_settings(
#     all_stats, categories=["Entropy", "MSE", "Height", "Coverage"]
# )


# pritn(all_stats)
plot_category_stats_by_e(all_stats, (4, "biased"))

setting_data = all_stats[
    (all_stats["Pairwise"] == "equal")
    & (all_stats["GaussianRadius"] == 4)
    & (all_stats["ErrorMargin"] == 0.0)
]
# print(setting_data)
