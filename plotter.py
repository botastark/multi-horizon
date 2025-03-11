import glob
import os
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse


import matplotlib.pyplot as plt
import pandas as pd
import glob


dir = "/home/bota/Desktop/active_sensing/"
# txt_dir = dir + "results/results_corner/txt/all"
txt_dir = dir + "results/gaussian_field_txts"


def get_closest_value(d, h):
    closest_key = min(d.keys(), key=lambda x: abs(x - h))
    return d[closest_key]


def parse_file_to_table(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse Pairwise and Gaussian radius
    strategy = None
    pairwise = None
    gaussian_radius = None
    error_margin = None
    confusion_matrix = None
    for line in lines:
        if line.startswith("Pairwise:"):
            pairwise = line.split(":")[1].strip()
        elif line.startswith("Strategy:"):
            strategy = line.split(":")[1].strip()
        elif line.startswith("Error margin:"):
            error_margin = line.split(":")[1].strip()
            if error_margin == "None":
                error_margin = 0.0
            else:
                error_margin = float(error_margin)
        elif "Gaussian radius" in line:
            parts = line.split()
            gaussian_radius = parts[-1]  # The last element is the radius
        elif "using orto" in line:
            gaussian_radius = "orto"
        elif "confision matrix" in line:
            confusion_matrix_str = line.split("matrix:")[1].strip()
            confusion_matrix_str = confusion_matrix_str.replace("] ", "],")

            # Remove curly braces and split into key-value pairs
            confusion_matrix_str = confusion_matrix_str.strip("{}")
            entries = confusion_matrix_str.split("],")

            # Parse into a dictionary
            confusion_matrix = {}
            for entry in entries:
                if not entry.strip():
                    continue
                key, value = entry.split(":")
                key = float(key.strip())
                value = [float(x) for x in value.strip(" []").split(",")]
                confusion_matrix[key] = value
            # print(f"file: {file_path}")
            # print(confusion_matrix.keys())
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
        sigma1, sigma2 = None, None
        # print(f"height: {height}")
        # print(f"confusion_matrix: {confusion_matrix}")
        if confusion_matrix is not None:
            [sigma1, sigma2] = get_closest_value(confusion_matrix, height)

        table_data.append([int(step), entropy, mse, height, coverage, sigma1, sigma2])

    df = pd.DataFrame(
        table_data,
        columns=["Step", "Entropy", "MSE", "Height", "Coverage", "sigma1", "sigma2"],
    )
    df["Strategy"] = strategy
    df["Pairwise"] = pairwise
    df["ErrorMargin"] = error_margin
    df["GaussianRadius"] = gaussian_radius
    df["sigma1"] = sigma1
    df["sigma2"] = sigma2
    df["Pairwise"] = df["Pairwise"].astype(str)
    df["GaussianRadius"] = df["GaussianRadius"].astype(str)
    df["ErrorMargin"] = df["ErrorMargin"].astype(float)

    return df


def aggregate_data_by_settings(folder_path):

    file_paths = glob.glob(f"{folder_path}/*.txt")
    assert (
        len(file_paths) != 0
    ), f"no file found with these settings in the folder:{folder_path}"
    # error_margin = extract_error_margin_from_folder(folder_path)
    all_data = [parse_file_to_table(file_path) for file_path in file_paths]
    # all_data = []
    # for file_path in file_paths:
    #     data = parse_file_to_table(file_path)  # Assuming this is a function you have
    #     data["ErrorMargin"] = error_margin  # Add the error margin to the data
    #     all_data.append(data)

    combined_df = pd.concat(all_data)
    return (
        combined_df.groupby(
            ["Strategy", "Pairwise", "GaussianRadius", "ErrorMargin", "Step"]
        )
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_category_stats_by_settings(stats, categories):

    radii = stats["GaussianRadius"].unique()
    pairwises = stats["Pairwise"].unique()
    error_margins = stats["ErrorMargin"].unique()
    strategy = stats["Strategy"].unique()
    print("strategies found: ", strategy)
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
            # ax[i].grid()
            ax[i].grid(True, linestyle="--", alpha=0.6, linewidth=0.7)  # Major grid
            ax[i].minorticks_on()  # Enable minor ticks
            ax[i].grid(True, which="minor", linestyle=":", alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        plt.show()


def plot_category_stats_by_e(stats, unique_settings, save_dir, show=False):
    plt.style.use("seaborn-v0_8-paper")  # Use a professional style
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    categories = ["Entropy", "MSE", "Height", "Coverage"]
    radius, pairwise = unique_settings
    if len(radius) == 1:
        radius = int(radius)
    error_margins = [0.0, 0.05, 0.1, 0.3]

    fig, ax = plt.subplots(
        len(categories), 1, figsize=(7, 3.5 * len(categories)), constrained_layout=True
    )
    colors = ["blue", "red", "green", "purple"]
    strategy = ""

    for iter, error_margin in enumerate(error_margins):
        if error_margin == 0.0:
            setting_data = stats[
                (stats["Pairwise"] == pairwise)
                & (stats["Strategy"] == "sweep")
                & (stats["GaussianRadius"] == radius)
                & (stats["ErrorMargin"] == error_margin)
            ]
        else:
            setting_data = stats[
                (stats["Pairwise"] == pairwise)
                & (stats["GaussianRadius"] == radius)
                & (stats["ErrorMargin"] == error_margin)
            ]

        steps = setting_data["Step"]

        if len(categories) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one category

        for i, category in enumerate(categories):
            mean_values = setting_data[(category, "mean")]
            std_values = setting_data[(category, "std")]

            ax[i].plot(
                steps,
                mean_values,
                label=f"E={error_margin}",
                color=colors[iter],
                linewidth=2,
            )
            ax[i].fill_between(
                steps,
                mean_values - std_values,
                mean_values + std_values,
                color=colors[iter],
                alpha=0.2,
            )

            ax[i].set_ylabel(category, fontsize=12)
            ax[i].tick_params(axis="both", labelsize=10)  # Increase tick label size

            # Remove unnecessary spines
            for spine in ["top", "right"]:
                ax[i].spines[spine].set_visible(False)

            # Enable minor ticks and grid on both axes
            ax[i].minorticks_on()
            ax[i].grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
            ax[i].grid(True, which="minor", linestyle=":", alpha=0.3, linewidth=0.5)

            # For subplots except the bottom one, hide only the x tick labels
            if i != len(categories) - 1:
                ax[i].set_xticklabels([])

    # Common X label
    fig.supxlabel("Steps", fontsize=12)

    # Improve legend placement
    ax[0].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=10
    )

    os.makedirs(save_dir, exist_ok=True)
    if (stats["Strategy"] == "sweep").any():
        strategy = "_sweep"

    plt.savefig(
        os.path.join(save_dir, f"plot{strategy}_r_{radius}_{pairwise}.png"),
        dpi=600,
        bbox_inches="tight",
    )

    print(f"Saving to {os.path.join(save_dir, f'plot_r_{radius}_{pairwise}.png')}")
    if show:
        plt.show()


# /home/bota/Desktop/active_sensing/results_orthomap/results_random/txt/equal_sweep_eNone_rorto


def plot_all_settings(stats, radius, save_dir, show=False):
    plt.style.use("seaborn-v0_8-paper")
    # Update global font settings for better visibility in a paper
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    categories = ["Entropy", "MSE", "Height", "Coverage"]
    pairwise_values = ["equal", "biased", "adaptive"]
    error_margins = [0.0, 0.05, 0.1, 0.3]
    colors = ["blue", "red", "green", "purple"]

    # Create grid: rows = categories, columns = pairwise settings
    fig, axes = plt.subplots(
        nrows=len(categories),
        ncols=len(pairwise_values),
        figsize=(7 * len(pairwise_values), 3.5 * len(categories)),
        constrained_layout=True,
    )

    if stats.empty:
        print("stats is empty ")
    else:
        print(f"stats: {stats.head()}")
    # Loop over each pairwise setting (columns) and category (rows)
    for col, pairwise_setting in enumerate(pairwise_values):
        for row, category in enumerate(categories):
            ax = axes[row, col]
            # Add caption with pairwise name only to the first row
            if row == 0:
                ax.set_title(pairwise_setting, fontsize=20, pad=10)
            for iter, error_margin in enumerate(error_margins):
                if error_margin == 0.0:
                    if radius == "4":
                        setting_data = stats[
                            (stats["Pairwise"] == pairwise_setting)
                            & (stats["Strategy"] == "ig")  # for gaussian
                            & (stats["GaussianRadius"] == radius)
                            & (stats["ErrorMargin"] == error_margin)
                        ]
                    else:
                        setting_data = stats[
                            (stats["Pairwise"] == pairwise_setting)
                            & (stats["Strategy"] == "sweep")  # for ortomap
                            & (stats["GaussianRadius"] == radius)
                            & (stats["ErrorMargin"] == error_margin)
                        ]

                else:
                    setting_data = stats[
                        (stats["Pairwise"] == pairwise_setting)
                        & (stats["GaussianRadius"] == radius)
                        & (stats["ErrorMargin"] == error_margin)
                    ]
                if setting_data.empty:
                    print(
                        f"setting_data is empty for {pairwise_setting}, {error_margin}"
                    )
                    break

                # Limit steps to 100
                setting_data = setting_data[setting_data["Step"] <= 100]
                steps = setting_data["Step"]

                mean_values = setting_data[(category, "mean")]
                std_values = setting_data[(category, "std")]
                label_text = "baseline" if error_margin == 0.0 else f"E={error_margin}"
                ax.plot(
                    steps,
                    mean_values,
                    label=label_text,
                    color=colors[iter],
                    linewidth=4,  # increased line width
                )
                ax.fill_between(
                    steps,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=colors[iter],
                    alpha=0.1,
                )

            # Only the leftmost column gets y-axis labels and tick labels
            if category == "Height":
                if radius == "4":
                    yticks = np.linspace(0, 5.4126 * 6, 7)
                else:
                    yticks = np.linspace(19.5, 19.5 + 7.79 * 5, 6)
                ax.set_yticks(yticks)

            elif category == "Entropy":
                num_ticks = 6  # e.g., 8 ticks including 0
                max_entropy = 175000 if radius == "4" else 7500
                yticks = np.linspace(0, max_entropy, num_ticks)
                ax.set_ylim(0, max_entropy)
                ax.set_yticks(yticks)

            elif category == "MSE":
                num_ticks = 5  # e.g., 5 ticks including 0
                max_mse = 0.32 if radius == "4" else 0.25
                yticks = np.round(np.linspace(0, max_mse, num_ticks), decimals=2)
                ax.set_ylim(0, max_mse)
                ax.set_yticks(yticks)

            if col == 0:

                if category == "Height":
                    ax.set_yticklabels([f"{ytick:.2f}" for ytick in yticks])
                    ax.set_ylabel(category)
                elif category == "Entropy":
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f"{x / 1e4:.1f}")
                    )
                    ax.set_ylabel(f"{category} (×1e4)")
                else:
                    ax.set_ylabel(category)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            # For bottom row, add x-axis label "Steps" for each column
            if row == len(categories) - 1:
                ax.set_xlabel("Steps", fontsize=20)
            ax.tick_params(axis="both", labelsize=20)
            ax.minorticks_on()
            ax.grid(True, linestyle="--", alpha=1, linewidth=1)
            ax.grid(True, which="minor", linestyle=":", alpha=0.7, linewidth=0.8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            # Hide x tick labels for all but the bottom row
            if row != len(categories) - 1:
                ax.set_xticklabels([])

    # For each row, set common y-axis limits across all columns
    for row in range(len(categories)):
        ymins = []
        ymaxs = []
        for col in range(len(pairwise_values)):
            ymin, ymax = axes[row, col].get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
        common_ymin = min(ymins)
        common_ymax = max(ymaxs)
        for col in range(len(pairwise_values)):
            axes[row, col].set_ylim(common_ymin, common_ymax)

    # Create a common legend using the handles from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=17,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"plot_all_settings_r_{radius}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saving to {out_path}")
    if show:
        plt.show()


def main(directory, show, pairwise, radius):

    if directory is None:
        directory = txt_dir
    all_stats = pd.DataFrame()

    # Aggregate data
    stats = aggregate_data_by_settings(directory)
    all_stats = pd.concat([all_stats, stats], ignore_index=True)

    # Print statistics
    print(f"unique error margins: {all_stats['ErrorMargin'].unique()}")
    print(f"unique rad: {all_stats['GaussianRadius'].unique()}")
    print(f"unique pairwise: {all_stats['Pairwise'].unique()}")

    plot_all_settings(all_stats, radius, os.path.join(dir, "plots"), show=show)
    # # Plot based on pairwise settings
    # pairwise_options = (
    #     ["equal", "biased", "adaptive"] if pairwise == "all" else [pairwise]
    # )

    # for factor in pairwise_options:
    #     plot_category_stats_by_e(
    #         all_stats,
    #         (radius, factor),
    #         os.path.join(dir, "plots"),
    #         show=show,
    #     )


#     # folders = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
#     # setting_data = all_stats[
#     #     (all_stats["Pairwise"] == "equal")
#     #     & (all_stats["GaussianRadius"] == 4)
#     #     & (all_stats["ErrorMargin"] == 0.0)
#     # ]
#     # print(setting_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot statistics from aggregated data."
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=None,
        help="Path to the directory containing data (optional)",
    )

    parser.add_argument(
        "--show", action="store_true", help="Whether to display the plots"
    )
    parser.add_argument(
        "--pairwise",
        type=str,
        default="all",
        choices=["all", "equal", "biased", "adaptive"],
        help="Choose a specific pairwise setting or 'all'",
    )

    parser.add_argument(
        "--radius",
        type=str,
        default="orto",
        help="Specify the radius setting (integer or 'orto')",
    )

    args = parser.parse_args()
    main(args.directory, args.show, args.pairwise, args.radius)
