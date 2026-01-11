#!/usr/bin/env python
"""
Create combined plot showing Force at 39mm Gape by all configuration parameters.

Generates a single figure with 4 subplots:
- By Hook Type
- By Hook Diameter
- By Mono Diameter
- By Crimp Type

Usage (from project root):
    python scripts/batch/plot_combined_force_39mm.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_and_merge_data():
    """Load batch metrics and merge with configuration data."""
    # Load batch metrics
    metrics_csv = Path("results/batch/batch_metrics_summary.csv")
    if not metrics_csv.exists():
        print(f"Error: {metrics_csv} not found. Run collect_batch_metrics.py first.")
        sys.exit(1)

    metrics_df = pd.read_csv(metrics_csv)

    # Load master datasheet for configuration info
    master_csv = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv")
    if not master_csv.exists():
        print(f"Error: {master_csv} not found.")
        sys.exit(1)

    master_df = pd.read_csv(master_csv)

    # Filter to only A-H batches and select configuration columns
    batch_config = master_df[master_df['Test #'].str.match(r'^[A-H]\d+', na=False)].copy()
    batch_config = batch_config[['Test #', 'Hook Type', 'Hook Diameter', 'Hook Size',
                                   'Mono Diameter', 'Weight Size', 'Crimp Type', 'Crimp Style']]
    batch_config.rename(columns={'Test #': 'Sample'}, inplace=True)

    # Merge metrics with configuration
    merged_df = pd.merge(metrics_df, batch_config, on='Sample', how='left')

    return merged_df


def calculate_stats_by_parameter(df, param_col):
    """Calculate statistics for a single parameter."""
    stats = []
    param_values = sorted(df[param_col].dropna().unique())

    for param_val in param_values:
        param_data = df[df[param_col] == param_val]['Force_at_39mm_Gape_N'].dropna()
        if len(param_data) > 0:
            stats.append({
                'Parameter': str(param_val),
                'Mean': param_data.mean(),
                'Std': param_data.std(),
                'Count': len(param_data)
            })

    return pd.DataFrame(stats)


def plot_combined_force_39mm(df, output_file):
    """
    Create combined plot with 4 subplots showing Force at 39mm by all parameters.

    Args:
        df: DataFrame with merged metrics and configuration data
        output_file: Path to save the plot
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Force at 39mm Gape: Configuration Parameter Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    # Define parameters and their properties
    params = [
        ('Hook Type', 'Hook Type (Round vs Forged)', ['#E63946', '#457B9D'], axes[0, 0]),
        ('Hook Diameter', 'Hook Diameter (mm)', ['#2E86AB', '#A23B72'], axes[0, 1]),
        ('Mono Diameter', 'Mono Diameter (mm)', ['#F18F01', '#C73E1D'], axes[1, 0]),
        ('Crimp Type', 'Crimp Type', ['#6A994E', '#BC4B51'], axes[1, 1])
    ]

    for param_col, title, colors, ax in params:
        # Calculate statistics
        stats_df = calculate_stats_by_parameter(df, param_col)

        if len(stats_df) == 0:
            continue

        # Create bar plot
        x_pos = np.arange(len(stats_df))
        bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                       capsize=8, alpha=0.85, color=colors[:len(stats_df)],
                       edgecolor='black', linewidth=1.5)

        # Customize subplot
        ax.set_xlabel(param_col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Force (N)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['Parameter'], fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

        # Set consistent y-axis limits
        ax.set_ylim([0, 1400])

        # Add value labels inside bars
        for bar, mean, std, count in zip(bars, stats_df['Mean'], stats_df['Std'], stats_df['Count']):
            height = bar.get_height()
            # Position text inside bar
            y_pos = height * 0.85 if height > 200 else height + std + 50
            v_align = 'top' if height > 200 else 'bottom'
            text_color = 'white' if height > 200 else 'black'

            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{mean:.1f} ± {std:.1f}\nn={count}',
                    ha='center', va=v_align, fontsize=11, fontweight='bold',
                    color=text_color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7) if height > 200 else None)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_combined_force_at_failure(df, output_file):
    """
    Create combined plot with 4 subplots showing Force at Failure by all parameters.

    Args:
        df: DataFrame with merged metrics and configuration data
        output_file: Path to save the plot
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Force at Failure: Configuration Parameter Comparison',
                 fontsize=18, fontweight='bold', y=0.995)

    # Define parameters and their properties
    params = [
        ('Hook Type', 'Hook Type (Round vs Forged)', ['#E63946', '#457B9D'], axes[0, 0]),
        ('Hook Diameter', 'Hook Diameter (mm)', ['#2E86AB', '#A23B72'], axes[0, 1]),
        ('Mono Diameter', 'Mono Diameter (mm)', ['#F18F01', '#C73E1D'], axes[1, 0]),
        ('Crimp Type', 'Crimp Type', ['#6A994E', '#BC4B51'], axes[1, 1])
    ]

    for param_col, title, colors, ax in params:
        # Calculate statistics for Force at Failure
        stats = []
        param_values = sorted(df[param_col].dropna().unique())

        for param_val in param_values:
            param_data = df[df[param_col] == param_val]['Force_at_Failure_N'].dropna()
            if len(param_data) > 0:
                stats.append({
                    'Parameter': str(param_val),
                    'Mean': param_data.mean(),
                    'Std': param_data.std(),
                    'Count': len(param_data)
                })

        stats_df = pd.DataFrame(stats)

        if len(stats_df) == 0:
            continue

        # Create bar plot
        x_pos = np.arange(len(stats_df))
        bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                       capsize=8, alpha=0.85, color=colors[:len(stats_df)],
                       edgecolor='black', linewidth=1.5)

        # Customize subplot
        ax.set_xlabel(param_col, fontsize=13, fontweight='bold')
        ax.set_ylabel('Force (N)', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats_df['Parameter'], fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

        # Set consistent y-axis limits
        ax.set_ylim([0, 2200])

        # Add value labels inside bars
        for bar, mean, std, count in zip(bars, stats_df['Mean'], stats_df['Std'], stats_df['Count']):
            height = bar.get_height()
            # Position text inside bar
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.85,
                    f'{mean:.0f} ± {std:.0f}\nn={count}',
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    print("=" * 80)
    print("Generating Combined Configuration Comparison Plots")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_and_merge_data()
    print(f"  Loaded {len(df)} samples")

    # Create output directory
    output_dir = Path("results/batch/configuration_analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating combined plots...")

    # Generate Force at 39mm combined plot
    output_file_39mm = output_dir / 'force_at_39mm_combined_parameters.png'
    plot_combined_force_39mm(df, output_file_39mm)

    # Generate Force at Failure combined plot
    output_file_failure = output_dir / 'force_at_failure_combined_parameters.png'
    plot_combined_force_at_failure(df, output_file_failure)

    print("\n" + "=" * 80)
    print("Combined Configuration Plots Complete")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - force_at_39mm_combined_parameters.png")
    print("  - force_at_failure_combined_parameters.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
