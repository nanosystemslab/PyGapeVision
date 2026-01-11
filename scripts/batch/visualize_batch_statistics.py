#!/usr/bin/env python
"""
Generate visualizations for batch statistics.

Creates plots comparing:
- Force at 39mm Gape across batches
- Force at Failure across batches
- Time to 39mm Gape across batches
- Time to Failure across batches
- Gape at Failure across batches
- Other key metrics

Usage (from project root):
    python scripts/batch/visualize_batch_statistics.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_batch_letter(sample_name):
    """Extract batch letter from sample name (e.g., 'A1' -> 'A')"""
    if isinstance(sample_name, str) and len(sample_name) > 0:
        return sample_name[0]
    return None


def plot_metric_by_batch(df, metric_col, title, ylabel, output_file, color='steelblue'):
    """
    Create a bar plot with error bars for a metric across batches.

    Args:
        df: DataFrame with Sample column and metric column
        metric_col: Name of the metric column
        title: Plot title
        ylabel: Y-axis label
        output_file: Path to save the plot
        color: Bar color
    """
    # Extract batch letter
    df['Batch'] = df['Sample'].apply(extract_batch_letter)

    # Calculate statistics by batch
    batch_stats = []
    batches = sorted(df['Batch'].unique())

    for batch in batches:
        if batch is None:
            continue
        batch_data = df[df['Batch'] == batch][metric_col].dropna()

        if len(batch_data) > 0:
            batch_stats.append({
                'Batch': batch,
                'Mean': batch_data.mean(),
                'Std': batch_data.std(),
                'Count': len(batch_data)
            })

    stats_df = pd.DataFrame(batch_stats)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(stats_df))
    bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                   capsize=5, alpha=0.8, color=color, edgecolor='black', linewidth=1.2)

    # Customize plot
    ax.set_xlabel('Batch', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['Batch'], fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for i, (bar, mean, std, count) in enumerate(zip(bars, stats_df['Mean'], stats_df['Std'], stats_df['Count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}\nn={count}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add legend showing error bars represent std dev
    legend_patch = mpatches.Patch(color='none', label='Error bars = ± 1 Std Dev')
    ax.legend(handles=[legend_patch], loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_multiple_metrics_comparison(df, metrics_info, output_file):
    """
    Create a multi-panel comparison plot for multiple metrics.

    Args:
        df: DataFrame with Sample column and metric columns
        metrics_info: List of tuples (metric_col, title, ylabel, color)
        output_file: Path to save the plot
    """
    # Extract batch letter
    df['Batch'] = df['Sample'].apply(extract_batch_letter)

    n_metrics = len(metrics_info)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 5*n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_col, title, ylabel, color) in zip(axes, metrics_info):
        # Calculate statistics by batch
        batch_stats = []
        batches = sorted(df['Batch'].unique())

        for batch in batches:
            if batch is None:
                continue
            batch_data = df[df['Batch'] == batch][metric_col].dropna()

            if len(batch_data) > 0:
                batch_stats.append({
                    'Batch': batch,
                    'Mean': batch_data.mean(),
                    'Std': batch_data.std(),
                    'Count': len(batch_data)
                })

        stats_df = pd.DataFrame(batch_stats)

        # Create the plot
        x_pos = np.arange(len(stats_df))
        bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                       capsize=5, alpha=0.8, color=color, edgecolor='black', linewidth=1.2)

        # Customize plot
        ax.set_xlabel('Batch', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['Batch'], fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on top of bars
        for bar, mean, count in zip(bars, stats_df['Mean'], stats_df['Count']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}\n(n={count})',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_box_plots_by_batch(df, metric_col, title, ylabel, output_file, color='lightblue'):
    """
    Create box plots for a metric across batches.

    Args:
        df: DataFrame with Sample column and metric column
        metric_col: Name of the metric column
        title: Plot title
        ylabel: Y-axis label
        output_file: Path to save the plot
        color: Box color
    """
    # Extract batch letter
    df['Batch'] = df['Sample'].apply(extract_batch_letter)

    # Prepare data for box plot
    batches = sorted(df['Batch'].unique())
    data_by_batch = []
    batch_labels = []

    for batch in batches:
        if batch is None:
            continue
        batch_data = df[df['Batch'] == batch][metric_col].dropna()
        if len(batch_data) > 0:
            data_by_batch.append(batch_data)
            batch_labels.append(f'{batch}\n(n={len(batch_data)})')

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    bp = ax.boxplot(data_by_batch, labels=batch_labels, patch_artist=True,
                     showmeans=True, meanline=False,
                     boxprops=dict(facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markeredgecolor='black', markersize=8))

    # Customize plot
    ax.set_xlabel('Batch (Sample Count)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    median_patch = mpatches.Patch(color='red', label='Median')
    mean_patch = mpatches.Patch(color='green', label='Mean')
    ax.legend(handles=[median_patch, mean_patch], loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 80)
    print("Generating Batch Statistics Visualizations")
    print("=" * 80)

    # Define paths
    metrics_csv = Path("results/batch/batch_metrics_summary.csv")
    output_dir = Path("results/batch/plots")

    if not metrics_csv.exists():
        print(f"\nError: Metrics file not found at {metrics_csv}")
        print("Please run collect_batch_metrics.py first.")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load batch metrics
    print(f"\nLoading batch metrics from: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    print(f"  Loaded {len(df)} samples\n")

    # Generate individual metric plots
    print("Generating bar plots with error bars...")

    plot_metric_by_batch(
        df, 'Force_at_39mm_Gape_N',
        'Force at 39mm Gape by Batch',
        'Force (N)',
        output_dir / 'force_at_39mm_by_batch.png',
        color='#2E86AB'
    )

    plot_metric_by_batch(
        df, 'Force_at_Failure_N',
        'Force at Failure by Batch',
        'Force (N)',
        output_dir / 'force_at_failure_by_batch.png',
        color='#A23B72'
    )

    plot_metric_by_batch(
        df, 'Time_to_39mm_Gape_s',
        'Time to 39mm Gape by Batch',
        'Time (s)',
        output_dir / 'time_to_39mm_by_batch.png',
        color='#F18F01'
    )

    plot_metric_by_batch(
        df, 'Time_to_Failure_s',
        'Time to Failure by Batch',
        'Time (s)',
        output_dir / 'time_to_failure_by_batch.png',
        color='#C73E1D'
    )

    plot_metric_by_batch(
        df, 'Gape_at_Failure_mm',
        'Gape at Failure by Batch',
        'Gape (mm)',
        output_dir / 'gape_at_failure_by_batch.png',
        color='#6A994E'
    )

    plot_metric_by_batch(
        df, 'Delta_Gape_at_Failure_mm',
        'Delta Gape at Failure by Batch',
        'Delta Gape (mm)',
        output_dir / 'delta_gape_at_failure_by_batch.png',
        color='#BC4B51'
    )

    # Generate box plots
    print("\nGenerating box plots...")

    plot_box_plots_by_batch(
        df, 'Force_at_39mm_Gape_N',
        'Force at 39mm Gape Distribution by Batch',
        'Force (N)',
        output_dir / 'force_at_39mm_boxplot.png',
        color='#A8DADC'
    )

    plot_box_plots_by_batch(
        df, 'Force_at_Failure_N',
        'Force at Failure Distribution by Batch',
        'Force (N)',
        output_dir / 'force_at_failure_boxplot.png',
        color='#F4A6D7'
    )

    plot_box_plots_by_batch(
        df, 'Time_to_Failure_s',
        'Time to Failure Distribution by Batch',
        'Time (s)',
        output_dir / 'time_to_failure_boxplot.png',
        color='#FFD6A5'
    )

    plot_box_plots_by_batch(
        df, 'Gape_at_Failure_mm',
        'Gape at Failure Distribution by Batch',
        'Gape (mm)',
        output_dir / 'gape_at_failure_boxplot.png',
        color='#B7E4C7'
    )

    # Generate combined comparison plot
    print("\nGenerating combined comparison plots...")

    metrics_info = [
        ('Force_at_39mm_Gape_N', 'Force at 39mm Gape by Batch', 'Force (N)', '#2E86AB'),
        ('Force_at_Failure_N', 'Force at Failure by Batch', 'Force (N)', '#A23B72'),
    ]

    plot_multiple_metrics_comparison(
        df, metrics_info,
        output_dir / 'force_comparison_combined.png'
    )

    print(f"\n{'=' * 80}")
    print(f"All plots saved to: {output_dir}/")
    print(f"{'=' * 80}")
    print("\nGenerated plots:")
    print("  Bar plots with error bars (Mean ± Std):")
    print("    - force_at_39mm_by_batch.png")
    print("    - force_at_failure_by_batch.png")
    print("    - time_to_39mm_by_batch.png")
    print("    - time_to_failure_by_batch.png")
    print("    - gape_at_failure_by_batch.png")
    print("    - delta_gape_at_failure_by_batch.png")
    print("\n  Box plots (showing distribution):")
    print("    - force_at_39mm_boxplot.png")
    print("    - force_at_failure_boxplot.png")
    print("    - time_to_failure_boxplot.png")
    print("    - gape_at_failure_boxplot.png")
    print("\n  Combined comparison:")
    print("    - force_comparison_combined.png")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
