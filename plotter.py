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

    Supports multiple experiments in one file separated by timestamp headers [YYYY-MM-DD...].

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    pd.DataFrame: DataFrame with parsed table data and added metadata columns.

    Raises:
    ValueError: If required metadata or the table header is not found.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Split file into separate experiments based on timestamp markers
    experiments = []
    current_experiment = []

    for line in lines:
        # Check if line is a timestamp marker (new experiment)
        if line.strip().startswith("[") and line.strip().endswith("]") and "T" in line:
            if current_experiment:
                experiments.append(current_experiment)
            current_experiment = [line]
        else:
            current_experiment.append(line)

    # Add the last experiment
    if current_experiment:
        experiments.append(current_experiment)

    # Parse each experiment separately
    all_dfs = []
    for exp_lines in experiments:
        df = _parse_single_experiment(exp_lines, file_path)
        if df is not None and not df.empty:
            all_dfs.append(df)

    # Combine all experiments
    if not all_dfs:
        raise ValueError(f"No valid experiments found in file: {file_path}")

    return pd.concat(all_dfs, ignore_index=True)


def _parse_single_experiment(lines, file_path):
    """
    Parse a single experiment from lines of text.

    Parameters:
    lines (list): Lines of text for one experiment.
    file_path (str): Original file path (for error messages).

    Returns:
    pd.DataFrame: Parsed experiment data, or None if parsing fails.
    """

    # Initialize metadata variables
    strategy = None
    pairwise = None
    gaussian_radius = None
    error_margin = None
    confusion_matrix = None
    n_agents = None
    iteration = None
    news_mode = None  # IG, IG_d, BS, BM

    # Extract metadata from the file lines
    for line in lines:
        if line.startswith("Pairwise:"):
            pairwise = line.split(":")[1].strip()
        elif line.startswith("Strategy:"):
            strategy = line.split(":")[1].strip()
        elif line.startswith("News mode:"):
            news_mode = line.split(":")[1].strip()
        elif line.startswith("Num agents:"):
            n_agents = int(line.split(":")[1].strip())
        elif line.startswith("Iteration:"):
            iteration = int(line.split(":")[1].strip())
        elif line.startswith("N agents:"):
            # Legacy support for old format
            n_agents = int(line.split(":")[1].strip())
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
        # Return None for incomplete experiments instead of raising error
        return None

    # Locate the starting point of the table
    start_index = None
    is_multi_agent = False
    for i, line in enumerate(lines):
        if line.strip().startswith("Step"):
            # Check if it's multi-agent format by looking for "Heights" in header
            if "Heights" in line and "Actions" in line:
                is_multi_agent = True
            # Check if next line is a separator (dashes)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                start_index = i + 2
            else:
                start_index = i + 1
            break

    if start_index is None:
        # Return None for experiments without data table
        return None

    # Process table rows
    table_data = []
    for line in lines[start_index:]:
        if not line.strip() or not line[0].isdigit():
            continue
        parts = line.split()

        if is_multi_agent:
            # Multi-agent format: Step Entropy MSE Coverage Heights[...] Actions[...] IGs[...]
            # Extract step, entropy, mse, coverage
            step = float(parts[0])
            entropy = float(parts[1])
            mse = float(parts[2])
            coverage = float(parts[3])

            # For multi-agent, we'll use average height from the list
            # Find the Heights list in the line
            heights_start = line.find("[", line.find(str(coverage)))
            if heights_start != -1:
                heights_end = line.find("]", heights_start)
                heights_str = line[heights_start + 1 : heights_end]
                heights = [
                    float(h.strip()) for h in heights_str.split(",") if h.strip()
                ]
                avg_height = sum(heights) / len(heights) if heights else 0.0
            else:
                avg_height = 0.0

            sigma1, sigma2 = None, None
            if confusion_matrix is not None:
                [sigma1, sigma2] = get_closest_value(confusion_matrix, avg_height)

            table_data.append(
                [int(step), entropy, mse, avg_height, coverage, sigma1, sigma2]
            )
        else:
            # Single-agent format: Step Entropy MSE Height Coverage Action IG
            step, entropy, mse, height, coverage = map(float, parts[:5])
            sigma1, sigma2 = None, None
            # If a confusion matrix is available, obtain the closest sigma values for the current height
            if confusion_matrix is not None:
                [sigma1, sigma2] = get_closest_value(confusion_matrix, height)

            table_data.append(
                [int(step), entropy, mse, height, coverage, sigma1, sigma2]
            )

    # Return None if no data rows found
    if not table_data:
        return None

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
    df["NumAgents"] = n_agents if n_agents is not None else 1
    df["Iteration"] = iteration if iteration is not None else 0
    df["NewsMode"] = (
        news_mode if news_mode is not None else "IG"
    )  # Default to IG (no sharing)
    df["sigma1"] = sigma1
    df["sigma2"] = sigma2
    # Ensure correct data types for consistency
    df["Pairwise"] = df["Pairwise"].astype(str)
    df["GaussianRadius"] = df["GaussianRadius"].astype(str)
    df["ErrorMargin"] = df["ErrorMargin"].astype(float)
    df["NewsMode"] = df["NewsMode"].astype(str)

    return df


def aggregate_data_by_settings(path):
    """
    Aggregate data from text files.

    This function reads .txt and .log files, parses each file into a DataFrame,
    concatenates them, and then groups the data by key settings to compute the mean and
    standard deviation for each metric.

    Multi-agent handling:
    - Entropy, MSE, Coverage are CUMULATIVE metrics from the fused belief across all agents
    - Height, Action, IG are per-agent values (logged as lists in multi-agent logs)
    - The plotter uses the cumulative/fused metrics for comparison

    Parameters:
    path (str): Path to a directory containing text files, or path to a single file.

    Returns:
    pd.DataFrame: DataFrame with aggregated statistics.

    Raises:
    FileNotFoundError: If no text files are found or file doesn't exist.
    """

    # Support both files and folders
    file_paths = []
    if os.path.isfile(path):
        file_paths = [path]
    elif os.path.isdir(path):
        file_paths = glob.glob(f"{path}/*.txt")
        file_paths.extend(glob.glob(f"{path}/*.log"))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    assert (
        len(file_paths) != 0
    ), f"no file found with these settings in the path: {path}"
    all_data = [parse_file_to_table(file_path) for file_path in file_paths]

    combined_df = pd.concat(all_data)

    # Determine grouping columns (include NewsMode and NumAgents if present)
    group_cols = ["Strategy", "Pairwise", "GaussianRadius", "ErrorMargin", "Step"]
    if "NewsMode" in combined_df.columns:
        group_cols.insert(4, "NewsMode")  # Add NewsMode before Step
    if "NumAgents" in combined_df.columns:
        group_cols.insert(1, "NumAgents")  # Add NumAgents after Strategy

    # Group data by settings and compute mean and standard deviation
    grouped = combined_df.groupby(group_cols).agg(["mean", "std"]).reset_index()

    return grouped


def plot_all_settings(stats, radius, save_dir, strategy=None, show=False):
    """
    Plot aggregated statistics for various settings.

    Creates a grid of plots for metrics including Entropy, MSE, Height, and Coverage over steps.
    Data is filtered based on the specified Gaussian radius and further subdivided by pairwise settings
    and error margins. The final plot is saved in the given directory.

    Parameters:
    stats (pd.DataFrame): Aggregated statistics DataFrame.
    radius (str): Gaussian radius setting to filter the data.
    save_dir (str): Directory to save the generated plot.
    strategy (str): Strategy name to include in filename (optional).
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
                # Filter data based on error margin and pairwise/radius settings
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
                # Let matplotlib auto-scale based on data, will be adjusted later
                pass

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
    filename = f"plot_all_settings_r_{radius}.png"
    if strategy:
        filename = f"plot_{strategy}_r_{radius}.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saving to {out_path}")
    if show:
        plt.show()


def plot_information_sharing_comparison(stats, radius, save_dir, show=False):
    """
    Plot comparison of information sharing modes (IG, IG_d, BS, BM) as in the paper.

    Creates a 2x2 or 3x2 grid:
    - Columns: IG (left) vs IG_d (right)
    - Rows: MSE evolution (R=∞), MSE evolution (R=5), local beliefs misalignment

    Paper reference: "Effect of information sharing in multi-agent scenarios"
    - IG: No information sharing (baseline)
    - IG_d: With position sharing (discounted)
    - BS: Single news belief shared to all neighbors
    - BM: Per-neighbor private news beliefs

    Parameters:
    stats (pd.DataFrame): Aggregated statistics DataFrame with NewsMode column.
    radius (str): Gaussian radius setting ('5' or 'orto').
    save_dir (str): Directory to save the generated plot.
    show (bool): If True, displays the plot interactively.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    # Define news modes and their display properties
    news_modes = {
        "IG": {"color": "blue", "linestyle": "-", "label": "IG (no sharing)"},
        "IG_d": {
            "color": "blue",
            "linestyle": "--",
            "label": "IG_d (position sharing)",
        },
        "BS": {"color": "red", "linestyle": "-", "label": "BS (single news)"},
        "BM": {"color": "green", "linestyle": "-", "label": "BM (per-neighbor news)"},
    }

    # Filter data for specified radius
    data = stats[stats["GaussianRadius"] == radius].copy()

    if data.empty:
        print(f"No data found for radius={radius}")
        return

    # Check available news modes
    available_modes = data["NewsMode"].unique() if "NewsMode" in data.columns else []
    print(f"Available news modes: {available_modes}")

    # Create figure: 2 columns (IG vs IG_d style), 2 rows (MSE, Coverage)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True)

    # Define what to plot
    metrics = ["MSE", "Coverage"]
    column_titles = ["IG (no position sharing)", "IG_d (with position sharing)"]

    # Mapping: left column shows IG and BS/BM without intent discount
    # right column shows IG_d variants
    left_modes = ["IG", "BS", "BM"]  # No position sharing
    right_modes = ["IG_d"]  # With position sharing (could add BS_d, BM_d if available)

    for col, (col_title, modes) in enumerate(
        zip(column_titles, [left_modes, right_modes])
    ):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]

            if row == 0:
                ax.set_title(col_title, fontsize=14, fontweight="bold")

            for mode in modes:
                if mode not in available_modes:
                    continue

                mode_data = data[data["NewsMode"] == mode]
                if mode_data.empty:
                    continue

                # Group by step and compute mean/std across iterations
                grouped = (
                    mode_data.groupby("Step")
                    .agg({(metric, "mean"): "mean", (metric, "std"): "mean"})
                    .reset_index()
                )

                if grouped.empty:
                    continue

                steps = grouped["Step"]
                mean_vals = grouped[(metric, "mean")]
                std_vals = grouped[(metric, "std")]

                props = news_modes.get(
                    mode, {"color": "gray", "linestyle": "-", "label": mode}
                )

                ax.plot(
                    steps,
                    mean_vals,
                    color=props["color"],
                    linestyle=props["linestyle"],
                    linewidth=2,
                    label=props["label"],
                )

                ax.fill_between(
                    steps,
                    mean_vals - std_vals,
                    mean_vals + std_vals,
                    color=props["color"],
                    alpha=0.15,
                )

            # Styling
            ax.set_ylabel(metric if col == 0 else "")
            if row == len(metrics) - 1:
                ax.set_xlabel("Steps")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.minorticks_on()
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            if col == 0:
                ax.legend(loc="best", framealpha=0.9)

    # Synchronize y-axis limits across columns for each metric
    for row in range(len(metrics)):
        ymin = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
        ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        for col in range(2):
            axes[row, col].set_ylim(ymin, ymax)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"info_sharing_comparison_r{radius}.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved information sharing comparison to {out_path}")

    if show:
        plt.show()


def plot_news_mode_comparison(stats, radius, save_dir, pairwise="equal", show=False):
    """
    Plot all news modes (IG, IG_d, BS, BM) on a single figure for comparison.

    This creates a simpler comparison showing MSE and Coverage evolution
    for all available information sharing modes.

    Parameters:
    stats (pd.DataFrame): Aggregated statistics DataFrame.
    radius (str): Gaussian radius setting.
    save_dir (str): Directory to save the plot.
    pairwise (str): Pairwise correlation type to filter by.
    show (bool): If True, displays the plot.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # News mode display properties
    mode_props = {
        "IG": {
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
            "label": "IG (no sharing)",
        },
        "IG_d": {
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
            "label": "IG_d (position)",
        },
        "BS": {
            "color": "#d62728",
            "linestyle": "-",
            "marker": "^",
            "label": "BS (broadcast)",
        },
        "BM": {
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "v",
            "label": "BM (per-neighbor)",
        },
    }

    # Filter data
    data = stats[
        (stats["GaussianRadius"] == radius) & (stats["Pairwise"] == pairwise)
    ].copy()

    if data.empty:
        print(f"No data for radius={radius}, pairwise={pairwise}")
        return

    # Get available news modes
    if "NewsMode" not in data.columns:
        print("NewsMode column not found in data")
        return

    available_modes = sorted(data["NewsMode"].unique())
    print(f"Plotting news modes: {available_modes}")

    # Create figure: 1 row, 2 columns (MSE, Coverage)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    metrics = [("MSE", "MSE"), ("Coverage", "Coverage")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for mode in available_modes:
            mode_data = data[data["NewsMode"] == mode]
            if mode_data.empty:
                continue

            # Limit to 100 steps
            mode_data = mode_data[mode_data["Step"] <= 100]

            steps = mode_data["Step"]
            mean_vals = mode_data[(metric, "mean")]
            std_vals = mode_data[(metric, "std")]

            props = mode_props.get(
                mode, {"color": "gray", "linestyle": "-", "marker": "", "label": mode}
            )

            # Plot with markers every 10 steps for visibility
            ax.plot(
                steps,
                mean_vals,
                color=props["color"],
                linestyle=props["linestyle"],
                linewidth=2,
                label=props["label"],
                markevery=10,
                marker=props.get("marker", ""),
                markersize=6,
            )

            ax.fill_between(
                steps,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=props["color"],
                alpha=0.15,
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} Evolution")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Information Sharing Comparison (r={radius}, {pairwise})",
        fontsize=16,
        fontweight="bold",
    )

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"news_mode_comparison_r{radius}_{pairwise}.png"
    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved news mode comparison to {out_path}")

    if show:
        plt.show()


def main(path, show, radius):
    """
    Main function to aggregate data from text files and generate plots.

    Parameters:
    path (str): Path to a directory containing data files or a single file.
    show (bool): Whether to display the plot interactively.
    radius (str): Gaussian radius setting for filtering the data.

    Raises:
    ValueError: If the path is not provided.
    """
    if path is None:
        raise ValueError("Path must be provided")

    all_stats = pd.DataFrame()

    # # Aggregate data
    stats = aggregate_data_by_settings(path)
    all_stats = pd.concat([all_stats, stats], ignore_index=True)
    # all_stats = aggregate_data_by_settings(path)

    # Print statistics
    print(f"unique strategies: {all_stats['Strategy'].unique()}")
    print(f"unique error margins: {all_stats['ErrorMargin'].unique()}")
    print(f"unique rad: {all_stats['GaussianRadius'].unique()}")
    print(f"unique pairwise: {all_stats['Pairwise'].unique()}")
    if "NewsMode" in all_stats.columns:
        print(f"unique news modes: {all_stats['NewsMode'].unique()}")

    # Determine strategy name (use first strategy if multiple exist)
    strategy = None
    if "Strategy" in all_stats.columns and len(all_stats["Strategy"].unique()) > 0:
        strategies = all_stats["Strategy"].unique()
        strategy = strategies[0] if len(strategies) == 1 else "multi_strategy"

    # Plot the aggregated statistics and save to the plots folder under the base directory
    save_dir = None
    if save_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(script_dir, "plots")
    plot_all_settings(all_stats, radius, save_dir, strategy=strategy, show=show)


def main_news_comparison(paths, show, radius, pairwise="equal"):
    """
    Main function to compare different news modes (IG, IG_d, BS, BM).

    Parameters:
    paths (list): List of paths to data directories for different news modes.
    show (bool): Whether to display the plot interactively.
    radius (str): Gaussian radius setting.
    pairwise (str): Pairwise correlation type.
    """
    all_stats = pd.DataFrame()

    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue
        try:
            stats = aggregate_data_by_settings(path)
            all_stats = pd.concat([all_stats, stats], ignore_index=True)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if all_stats.empty:
        print("No data found!")
        return

    # Print available data - handle MultiIndex columns from aggregation
    print(f"\n=== Data Summary ===")
    print(f"Strategies: {all_stats['Strategy'].unique()}")
    print(
        f"News modes: {all_stats['NewsMode'].unique() if 'NewsMode' in all_stats.columns else 'N/A'}"
    )
    print(f"Pairwise: {all_stats['Pairwise'].unique()}")
    print(f"Radius: {all_stats['GaussianRadius'].unique()}")
    # NumAgents may not be in grouping columns, so check if it exists
    if "NumAgents" in all_stats.columns:
        print(f"Num agents: {all_stats['NumAgents'].unique()}")

    # Save directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(script_dir, "plots")

    # Generate comparison plots
    if "NewsMode" in all_stats.columns:
        plot_news_mode_comparison(
            all_stats, radius, save_dir, pairwise=pairwise, show=show
        )
        plot_information_sharing_comparison(all_stats, radius, save_dir, show=show)
    else:
        print("NewsMode column not found - using standard plotting")
        plot_all_settings(all_stats, radius, save_dir, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot statistics from aggregated data."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to a directory containing data files or a single file (optional)",
    )

    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        default=None,
        help="Multiple paths for news mode comparison (e.g., --paths dir1 dir2 dir3)",
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

    parser.add_argument(
        "--pairwise",
        type=str,
        default="equal",
        help="Pairwise correlation type for comparison plots",
    )

    parser.add_argument(
        "--compare-news",
        action="store_true",
        help="Generate information sharing comparison plots (IG, IG_d, BS, BM)",
    )

    args = parser.parse_args()

    if args.compare_news and args.paths:
        # Use multiple paths for news mode comparison
        main_news_comparison(args.paths, args.show, args.radius, args.pairwise)
    elif args.compare_news and args.path:
        # Use single path but generate news comparison plots
        main_news_comparison([args.path], args.show, args.radius, args.pairwise)
    else:
        # Standard plotting
        main(args.path, args.show, args.radius)
