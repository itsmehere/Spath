"""
Publication-quality plots for Experiment 2 using Seaborn.
Generates figures suitable for conference papers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-ready style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.4)

# Color palette - colorblind friendly
COLORS = {
    "0-shot": "#E69F00",  # Orange
    "1-shot": "#56B4E9",  # Sky blue
    "3-shot": "#009E73",  # Teal
    "5-shot": "#CC79A7",  # Pink
    "Baseline": "#0072B2",  # Blue
    "Baseline (Shuffled)": "#D55E00",  # Vermilion
}

# Line styles for differentiation
LINE_STYLES = {
    "0-shot": "-",
    "1-shot": "-",
    "3-shot": "-",
    "5-shot": "-",
    "Baseline": "--",
    "Baseline (Shuffled)": ":",
}

# Figure sizes (width, height) in inches - suitable for two-column papers
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0
FIG_HEIGHT = 2.8

# Output directory
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_csv_data(filepath: Path) -> pd.DataFrame:
    """Load CSV and clean column names."""
    df = pd.read_csv(filepath)
    return df


def extract_value_column(df: pd.DataFrame, metric_keyword: str) -> tuple[str, str, str]:
    """Find the main value column and its MIN/MAX columns."""
    cols = df.columns.tolist()
    value_col = None
    for col in cols:
        if metric_keyword in col and "__MIN" not in col and "__MAX" not in col:
            if col not in ["Step", "train/global_step"]:
                value_col = col
                break

    if value_col is None:
        # Try to find any column that's not Step or step-related
        for col in cols:
            if col not in ["Step", "train/global_step"] and "__MIN" not in col and "__MAX" not in col:
                if "_step" not in col.lower():
                    value_col = col
                    break

    min_col = f"{value_col}__MIN" if value_col else None
    max_col = f"{value_col}__MAX" if value_col else None

    return value_col, min_col, max_col


def get_step_column(df: pd.DataFrame) -> str:
    """Get the step/x-axis column name."""
    if "Step" in df.columns:
        return "Step"
    elif "train/global_step" in df.columns:
        return "train/global_step"
    return df.columns[0]


def load_accuracy_data(include_baseline_shuffle: bool = True) -> dict[str, pd.DataFrame]:
    """Load all accuracy CSVs."""
    accuracy_dir = BASE_DIR / "eval_accuracy"
    data = {}

    files = {
        "0-shot": "0_shot_accuracy.csv",
        "1-shot": "1_shot_accuracy.csv",
        "3-shot": "3_shot_accuracy.csv",
        "5-shot": "5_shot_accuracy.csv",
    }
    if include_baseline_shuffle:
        files["Baseline (Shuffled)"] = "baseline_shuffle_accuracy.csv"

    for label, filename in files.items():
        filepath = accuracy_dir / filename
        if filepath.exists():
            data[label] = load_csv_data(filepath)

    return data


def load_eval_loss_data(include_baseline_shuffle: bool = True) -> dict[str, pd.DataFrame]:
    """Load all eval loss CSVs."""
    loss_dir = BASE_DIR / "eval_loss"
    data = {}

    files = {
        "1-shot": "1_shot_eval_loss.csv",
        "3-shot": "3_shot_eval_loss.csv",
        "5-shot": "5_shot_eval_loss.csv",
        "Baseline": "baseline_eval_loss.csv",
    }
    if include_baseline_shuffle:
        files["Baseline (Shuffled)"] = "baseline_shuffle_eval_loss.csv"

    for label, filename in files.items():
        filepath = loss_dir / filename
        if filepath.exists():
            data[label] = load_csv_data(filepath)

    return data


def load_mae_data(include_baseline_shuffle: bool = True) -> dict[str, pd.DataFrame]:
    """Load all MAE CSVs."""
    mae_dir = BASE_DIR / "mae"
    data = {}

    files = {
        "0-shot": "0_shot_mae.csv",
        "1-shot": "1_shot_mae.csv",
        "3-shot": "3_shot_mae.csv",
        "5-shot": "5_shot_mae.csv",
        "Baseline": "baseline_mae.csv",
    }
    if include_baseline_shuffle:
        files["Baseline (Shuffled)"] = "baseline_shuffle_mae.csv"

    for label, filename in files.items():
        filepath = mae_dir / filename
        if filepath.exists():
            data[label] = load_csv_data(filepath)

    return data


def load_train_loss_data(include_baseline_shuffle: bool = True) -> dict[str, pd.DataFrame]:
    """Load all train loss CSVs."""
    loss_dir = BASE_DIR / "train_loss"
    data = {}

    files = {
        "1-shot": "1_shot_train_loss.csv",
        "3-shot": "3_shot_train_loss.csv",
        "5-shot": "5_shot_train_loss.csv",
        "Baseline": "baseline_train_loss.csv",
    }
    if include_baseline_shuffle:
        files["Baseline (Shuffled)"] = "baseline_shuffle_train_loss.csv"

    for label, filename in files.items():
        filepath = loss_dir / filename
        if filepath.exists():
            data[label] = load_csv_data(filepath)

    return data


def compute_binomial_ci(p: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Compute 95% confidence interval for binomial proportion using Wilson score interval."""
    # Wilson score interval (better for proportions near 0 or 1)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return np.maximum(0, center - margin), np.minimum(1, center + margin)


def plot_metric(
    data: dict[str, pd.DataFrame],
    metric_keyword: str,
    ylabel: str,
    title: str,
    output_name: str,
    figsize: tuple[float, float] = (DOUBLE_COL_WIDTH, FIG_HEIGHT),
    show_confidence: bool = True,
    alpha_fill: float = 0.15,
    subsample: int = 1,
    legend_loc: str = "best",
    ylim: tuple[float, float] | None = None,
    clip_outliers: bool = False,
    clip_percentile: float = 99.5,
    binomial_n: int | None = None,
) -> None:
    """Create a publication-quality line plot for a metric.
    
    Args:
        binomial_n: If provided, compute binomial confidence intervals assuming
                   this many samples per data point (for accuracy/proportion data).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, df in data.items():
        step_col = get_step_column(df)
        value_col, min_col, max_col = extract_value_column(df, metric_keyword)

        if value_col is None:
            print(f"Warning: Could not find value column for {label}")
            continue

        # Subsample data for cleaner plots
        df_plot = df.iloc[::subsample].copy()

        x = df_plot[step_col].values
        y = df_plot[value_col].values

        # Clip outliers if requested
        if clip_outliers:
            threshold = np.percentile(y[np.isfinite(y)], clip_percentile)
            y = np.clip(y, None, threshold)

        color = COLORS.get(label, "#333333")
        linestyle = LINE_STYLES.get(label, "-")

        # Plot main line
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            marker="",
        )

        # Add confidence band
        if show_confidence:
            if binomial_n is not None:
                # Compute binomial confidence interval
                y_min, y_max = compute_binomial_ci(y, binomial_n)
                ax.fill_between(x, y_min, y_max, color=color, alpha=alpha_fill)
            elif min_col in df_plot.columns and max_col in df_plot.columns:
                # Use provided MIN/MAX columns
                y_min = df_plot[min_col].values
                y_max = df_plot[max_col].values
                # Only show fill if there's actual variance
                if not (y_min == y_max).all():
                    ax.fill_between(x, y_min, y_max, color=color, alpha=alpha_fill)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    if ylim:
        ax.set_ylim(ylim)

    # Format legend
    ax.legend(
        loc=legend_loc,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=9,
        ncol=2 if len(data) > 4 else 1,
    )

    # Style improvements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(OUTPUT_DIR / f"{output_name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_name}.pdf, {output_name}.png")


def plot_combined_figure() -> None:
    """Create a 2x2 grid of all metrics for a compact overview figure."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.5))

    # Load all data (excluding baseline shuffle)
    accuracy_data = load_accuracy_data(include_baseline_shuffle=False)
    eval_loss_data = load_eval_loss_data(include_baseline_shuffle=False)
    mae_data = load_mae_data(include_baseline_shuffle=False)
    train_loss_data = load_train_loss_data(include_baseline_shuffle=False)

    # (ax, data, metric_keyword, ylabel, title, ylim, binomial_n)
    plot_configs = [
        (axes[0, 0], accuracy_data, "accuracy", "Eval Accuracy", "(a) Evaluation Accuracy", None, 85),
        (axes[0, 1], eval_loss_data, "loss", "Cross-Entropy", "(b) Eval Cross-Entropy Loss", None, None),
        (axes[1, 0], mae_data, "mae", "MAE", "(c) Mean Absolute Error", (0, 15), None),
        (axes[1, 1], train_loss_data, "loss", "Cross-Entropy", "(d) Train Cross-Entropy Loss", None, None),
    ]

    for ax, data, metric_keyword, ylabel, title, ylim, binomial_n in plot_configs:
        for label, df in data.items():
            step_col = get_step_column(df)
            value_col, min_col, max_col = extract_value_column(df, metric_keyword)

            if value_col is None:
                continue

            # Subsample for cleaner combined plot
            df_plot = df.iloc[::2].copy()
            x = df_plot[step_col].values
            y = df_plot[value_col].values

            color = COLORS.get(label, "#333333")
            linestyle = LINE_STYLES.get(label, "-")

            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.2)

            # Add error bars for accuracy
            if binomial_n is not None:
                y_min, y_max = compute_binomial_ci(y, binomial_n)
                ax.fill_between(x, y_min, y_max, color=color, alpha=0.15)

        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Create a shared legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Combine with other plots' labels to get complete set
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for handle, lbl in zip(h, l):
                if lbl not in labels:
                    handles.append(handle)
                    labels.append(lbl)

    # Sort legend entries
    order = ["0-shot", "1-shot", "3-shot", "5-shot", "Baseline"]
    sorted_pairs = sorted(
        zip(handles, labels), key=lambda x: order.index(x[1]) if x[1] in order else len(order)
    )
    handles, labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    fig.savefig(OUTPUT_DIR / "combined_metrics.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "combined_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: combined_metrics.pdf, combined_metrics.png")


def plot_accuracy_comparison() -> None:
    """Create accuracy comparison with focus on best performing methods."""
    data = load_accuracy_data(include_baseline_shuffle=False)
    plot_metric(
        data=data,
        metric_keyword="accuracy",
        ylabel="Evaluation Accuracy",
        title="Evaluation Accuracy Across Training",
        output_name="eval_accuracy",
        legend_loc="lower right",
        ylim=(0, None),
        binomial_n=85,  # 85 samples per accuracy calculation
    )


def plot_eval_loss_comparison() -> None:
    """Create evaluation loss comparison."""
    data = load_eval_loss_data(include_baseline_shuffle=False)
    plot_metric(
        data=data,
        metric_keyword="loss",
        ylabel="Cross-Entropy Loss",
        title="Evaluation Cross-Entropy Loss Across Training",
        output_name="eval_loss",
        legend_loc="upper right",
    )


def plot_mae_comparison() -> None:
    """Create MAE comparison."""
    data = load_mae_data(include_baseline_shuffle=False)
    plot_metric(
        data=data,
        metric_keyword="mae",
        ylabel="Mean Absolute Error",
        title="Mean Absolute Error Across Training",
        output_name="mae",
        legend_loc="upper right",
        ylim=(0, 15),  # Set reasonable y-limit based on typical MAE values
    )


def plot_train_loss_comparison() -> None:
    """Create training loss comparison."""
    data = load_train_loss_data(include_baseline_shuffle=False)
    plot_metric(
        data=data,
        metric_keyword="loss",
        ylabel="Cross-Entropy Loss",
        title="Training Cross-Entropy Loss Across Training",
        output_name="train_loss",
        subsample=10,  # Subsample heavily for train loss (many points)
        legend_loc="upper right",
    )


def plot_few_shot_scaling() -> None:
    """Create a focused comparison showing few-shot scaling effect."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, FIG_HEIGHT))

    # Accuracy scaling
    accuracy_data = load_accuracy_data(include_baseline_shuffle=False)
    few_shot_labels = ["0-shot", "1-shot", "3-shot", "5-shot"]

    for label in few_shot_labels:
        if label not in accuracy_data:
            continue
        df = accuracy_data[label]
        step_col = get_step_column(df)
        value_col, _, _ = extract_value_column(df, "accuracy")
        if value_col:
            x = df[step_col].values
            y = df[value_col].values
            color = COLORS.get(label)
            ax1.plot(x, y, label=label, color=color, linewidth=1.8)
            # Add error bars
            y_min, y_max = compute_binomial_ci(y, n=85)
            ax1.fill_between(x, y_min, y_max, color=color, alpha=0.15)

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Evaluation Accuracy", fontsize=11)
    ax1.set_title("(a) Accuracy vs. Few-Shot Examples", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.3)

    # MAE scaling
    mae_data = load_mae_data()
    for label in few_shot_labels:
        if label not in mae_data:
            continue
        df = mae_data[label]
        step_col = get_step_column(df)
        value_col, _, _ = extract_value_column(df, "mae")
        if value_col:
            ax2.plot(
                df[step_col],
                df[value_col],
                label=label,
                color=COLORS.get(label),
                linewidth=1.8,
            )

    ax2.set_ylim(0, 15)  # Set reasonable y-limit
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Mean Absolute Error", fontsize=11)
    ax2.set_title("(b) MAE vs. Few-Shot Examples", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "few_shot_scaling.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "few_shot_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: few_shot_scaling.pdf, few_shot_scaling.png")


def plot_baseline_comparison() -> None:
    """Create individual comparison plots between Baseline and Baseline (Shuffled)."""
    baseline_labels = ["Baseline", "Baseline (Shuffled)"]

    # Load all data and filter for baselines only
    eval_loss_data = {k: v for k, v in load_eval_loss_data().items() if k in baseline_labels}
    mae_data = {k: v for k, v in load_mae_data().items() if k in baseline_labels}
    train_loss_data = {k: v for k, v in load_train_loss_data().items() if k in baseline_labels}

    # Individual plots for each metric
    plot_configs = [
        (eval_loss_data, "loss", "Cross-Entropy Loss", "Baseline vs. Shuffled: Eval Cross-Entropy", "baseline_eval_loss", 1),
        (mae_data, "mae", "Mean Absolute Error", "Baseline vs. Shuffled: MAE", "baseline_mae", 1),
        (train_loss_data, "loss", "Cross-Entropy Loss", "Baseline vs. Shuffled: Train Cross-Entropy", "baseline_train_loss", 10),
    ]

    for data, metric_keyword, ylabel, title, output_name, subsample in plot_configs:
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH * 1.5, FIG_HEIGHT))

        # Find max x value from Baseline (Shuffled) to limit x-axis
        max_x = None
        if "Baseline (Shuffled)" in data:
            shuffle_df = data["Baseline (Shuffled)"]
            step_col = get_step_column(shuffle_df)
            max_x = shuffle_df[step_col].max()

        for label, df in data.items():
            step_col = get_step_column(df)
            value_col, _, _ = extract_value_column(df, metric_keyword)

            if value_col is None:
                continue

            df_plot = df.iloc[::subsample].copy()

            # Filter to max_x if available
            if max_x is not None:
                df_plot = df_plot[df_plot[step_col] <= max_x]

            x = df_plot[step_col].values
            y = df_plot[value_col].values

            color = COLORS.get(label, "#333333")
            linestyle = LINE_STYLES.get(label, "-")

            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0)

        if max_x is not None:
            ax.set_xlim(0, max_x)

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.legend(frameon=True, fontsize=10, loc="best")

        plt.tight_layout()

        fig.savefig(OUTPUT_DIR / f"{output_name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_name}.pdf, {output_name}.png")


def main() -> None:
    """Generate all publication-quality plots."""
    print(f"Generating plots in: {OUTPUT_DIR}")
    print("-" * 50)

    # Individual metric plots
    plot_accuracy_comparison()
    plot_eval_loss_comparison()
    plot_mae_comparison()
    plot_train_loss_comparison()

    # Combined overview figure
    plot_combined_figure()

    # Few-shot scaling analysis
    plot_few_shot_scaling()

    # Baseline comparison
    plot_baseline_comparison()

    print("-" * 50)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
