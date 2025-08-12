import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

dir = None


def get_closest_value(d, h):
    closest_key = min(d.keys(), key=lambda x: abs(x - h))
    return d[closest_key]


def parse_file_to_table(file_path):
    """
    Parse a text file into a DataFrame containing both metadata and table data.

    The text file should include metadata (e.g., 'Pairwise', 'Strategy', 'Error margin',
    'Gaussian radius', and optionally a 'confision matrix') as well as a table starting
    with a header that begins with "Step".

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    pd.DataFrame: DataFrame with parsed table data and added metadata columns.

    Raises:
    ValueError: If required metadata or the table header is not found.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Initialize metadata variables
    strategy = None
    pairwise = None
    gaussian_radius = None
    error_margin = None
    confusion_matrix = None
    # Extract metadata from the file lines
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
        # Ensure necessary metadata exists
    if pairwise is None or gaussian_radius is None:
        raise ValueError(
            f"Required metadata (Pairwise: {pairwise}, Error Margin: {error_margin}, Gaussian radius: {gaussian_radius}) not found in file: {file_path}"
        )

    # Locate the starting point of the table
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Step"):
            start_index = i + 2
            break

    if start_index is None:
        raise ValueError(f"Table header 'Step' not found in the file: {file_path}")
    # Process table rows
    table_data = []
    for line in lines[start_index:]:
        if not line.strip() or not line[0].isdigit():
            continue
        parts = line.split()
        step, entropy, mse, height, coverage = map(float, parts[:5])
        sigma1, sigma2 = None, None
        # If a confusion matrix is available, obtain the closest sigma values for the current height
        if confusion_matrix is not None:
            [sigma1, sigma2] = get_closest_value(confusion_matrix, height)

        table_data.append([int(step), entropy, mse, height, coverage, sigma1, sigma2])
    # Create DataFrame from the table data
    df = pd.DataFrame(
        table_data,
        columns=["Step", "Entropy", "MSE", "Height", "Coverage", "sigma1", "sigma2"],
    )
    # Add metadata columns to the DataFrame
    df["Strategy"] = strategy
    df["Pairwise"] = pairwise
    df["ErrorMargin"] = error_margin
    df["GaussianRadius"] = gaussian_radius
    df["sigma1"] = sigma1
    df["sigma2"] = sigma2
    # Ensure correct data types for consistency
    df["Pairwise"] = df["Pairwise"].astype(str)
    df["GaussianRadius"] = df["GaussianRadius"].astype(str)
    df["ErrorMargin"] = df["ErrorMargin"].astype(float)

    return df


def aggregate_data_by_settings(folder_path):
    """
    Aggregate data from all text files in the given folder.

    This function reads all .txt files from the folder, parses each file into a DataFrame,
    concatenates them, and then groups the data by key settings to compute the mean and
    standard deviation for each metric.

    Parameters:
    folder_path (str): Path to the directory containing text files.

    Returns:
    pd.DataFrame: DataFrame with aggregated statistics.

    Raises:
    FileNotFoundError: If no text files are found in the specified folder.
    """

    file_paths = glob.glob(f"{folder_path}/*.txt")
    assert (
        len(file_paths) != 0
    ), f"no file found with these settings in the folder:{folder_path}"
    all_data = [parse_file_to_table(file_path) for file_path in file_paths]

    combined_df = pd.concat(all_data)

    # Group data by settings and compute mean and standard deviation
    grouped = (
        combined_df.groupby(
            ["Strategy", "Pairwise", "GaussianRadius", "ErrorMargin", "Step"]
        )
        .agg(["mean", "std"])
        .reset_index()
    )

    return grouped


def plot_all_settings(stats, radius, save_dir, show=False):
    """
    Plot aggregated statistics for various settings.

    Creates a grid of plots for metrics including Entropy, MSE, Height, and Coverage over steps.
    Data is filtered based on the specified Gaussian radius and further subdivided by pairwise settings
    and error margins. The final plot is saved in the given directory.

    Parameters:
    stats (pd.DataFrame): Aggregated statistics DataFrame.
    radius (str): Gaussian radius setting to filter the data.
    save_dir (str): Directory to save the generated plot.
    show (bool): If True, displays the plot interactively.
    """
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
        print(f"Stats preview:\n{stats.head()}")

    # Loop over each pairwise setting (columns) and category (rows)
    for col, pairwise_setting in enumerate(pairwise_values):
        for row, category in enumerate(categories):
            ax = axes[row, col]
            # Add caption with pairwise name only to the first row
            if row == 0:
                ax.set_title(pairwise_setting, fontsize=20, pad=10)
            for iter, error_margin in enumerate(error_margins):
                # Filter data based on error margin and strategy/radius settings
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
                        f"No data for {pairwise_setting} with error margin {error_margin}."
                    )
                    continue

                # Limit to steps ≤ 100 for clarity in plots
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


def main(directory, show, radius):
    """
    Main function to aggregate data from text files and generate plots.

    Parameters:
    directory (str): Path to the directory containing data files.
    show (bool): Whether to display the plot interactively.
    radius (str): Gaussian radius setting for filtering the data.

    Raises:
    ValueError: If the directory is not provided.
    """
    if directory is None:
        raise ValueError("Directory must be provided")

    all_stats = pd.DataFrame()

    # # Aggregate data
    stats = aggregate_data_by_settings(directory)
    all_stats = pd.concat([all_stats, stats], ignore_index=True)
    # all_stats = aggregate_data_by_settings(directory)

    # Print statistics
    print(f"unique error margins: {all_stats['ErrorMargin'].unique()}")
    print(f"unique rad: {all_stats['GaussianRadius'].unique()}")
    print(f"unique pairwise: {all_stats['Pairwise'].unique()}")
    # Plot the aggregated statistics and save to the plots folder under the base directory
    save_dir = None
    if save_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(script_dir, "plots")
    plot_all_settings(all_stats, radius, save_dir, show=show)


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
        "--radius",
        type=str,
        default="orto",
        help="Specify the radius setting (integer or 'orto')",
    )

    args = parser.parse_args()
    main(args.directory, args.show, args.radius)
