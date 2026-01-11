#!/usr/bin/env python
"""
Create comprehensive summary plots showing Force vs All Configuration Parameters.

Generates plots with all configuration combinations on x-axis and force on y-axis.

Usage (from project root):
    python scripts/batch/plot_force_summary.py
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


def create_config_label(row):
    """Create a concise configuration label from row data."""
    hook_type = str(row['Hook Type'])[:1]  # R or F
    hook_diam = str(row['Hook Diameter'])
    mono_diam = str(row['Mono Diameter'])
    crimp = str(row['Crimp Type']).replace('Jap ', '')

    return f"{hook_type}{hook_diam}/{mono_diam}\n{crimp}"


def plot_comprehensive_force_summary(df, output_dir):
    """
    Create comprehensive plot showing Force at Failure for all configuration combinations.

    Args:
        df: DataFrame with merged metrics and configuration data
        output_dir: Directory to save plots
    """
    # Create configuration label
    df['Config_Label'] = df.apply(create_config_label, axis=1)

    # Group by configuration and calculate statistics
    config_groups = df.groupby(['Hook Type', 'Hook Diameter', 'Mono Diameter', 'Crimp Type'])

    stats = []
    for config, group in config_groups:
        force_data = group['Force_at_Failure_N'].dropna()
        if len(force_data) > 0:
            config_label = create_config_label(group.iloc[0])
            stats.append({
                'Config': config_label,
                'Hook Type': config[0],
                'Hook Diameter': config[1],
                'Mono Diameter': config[2],
                'Crimp Type': config[3],
                'Mean': force_data.mean(),
                'Std': force_data.std(),
                'Count': len(force_data)
            })

    stats_df = pd.DataFrame(stats)

    # Sort by mean force for better visualization
    stats_df = stats_df.sort_values('Mean', ascending=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Color by crimp type
    colors = []
    for crimp in stats_df['Crimp Type']:
        if crimp == 'Jap C':
            colors.append('#3498db')  # Blue
        else:
            colors.append('#e74c3c')  # Red

    x_pos = np.arange(len(stats_df))
    bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                   capsize=5, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Configuration (Hook Type, Diameter / Mono Diameter, Crimp Type)',
                   fontsize=14, fontweight='bold')
    ax.set_ylabel('Force at Failure (N)', fontsize=14, fontweight='bold')
    ax.set_title('Force at Failure: Comprehensive Configuration Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['Config'], fontsize=9, rotation=0)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels inside bars
    for bar, mean, count in zip(bars, stats_df['Mean'], stats_df['Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{mean:.0f}\n(n={count})',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Jap C Crimp'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Jap D Crimp')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    # Add horizontal line at mean
    overall_mean = df['Force_at_Failure_N'].mean()
    ax.axhline(y=overall_mean, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Overall Mean: {overall_mean:.0f} N')

    plt.tight_layout()
    output_file = output_dir / 'force_at_failure_all_configurations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_comprehensive_force_at_39mm_summary(df, output_dir):
    """
    Create comprehensive plot showing Force at 39mm Gape for all configuration combinations.

    Args:
        df: DataFrame with merged metrics and configuration data
        output_dir: Directory to save plots
    """
    # Create configuration label
    df['Config_Label'] = df.apply(create_config_label, axis=1)

    # Group by configuration and calculate statistics
    config_groups = df.groupby(['Hook Type', 'Hook Diameter', 'Mono Diameter', 'Crimp Type'])

    stats = []
    for config, group in config_groups:
        force_data = group['Force_at_39mm_Gape_N'].dropna()
        if len(force_data) > 0:
            config_label = create_config_label(group.iloc[0])
            stats.append({
                'Config': config_label,
                'Hook Type': config[0],
                'Hook Diameter': config[1],
                'Mono Diameter': config[2],
                'Crimp Type': config[3],
                'Mean': force_data.mean(),
                'Std': force_data.std(),
                'Count': len(force_data)
            })

    stats_df = pd.DataFrame(stats)

    # Sort by mean force
    stats_df = stats_df.sort_values('Mean', ascending=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Color by hook diameter
    colors = []
    for diam in stats_df['Hook Diameter']:
        if diam == 4.2:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#9b59b6')  # Purple

    x_pos = np.arange(len(stats_df))
    bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                   capsize=5, alpha=0.85, color=colors, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Configuration (Hook Type, Diameter / Mono Diameter, Crimp Type)',
                   fontsize=14, fontweight='bold')
    ax.set_ylabel('Force at 39mm Gape (N)', fontsize=14, fontweight='bold')
    ax.set_title('Force at 39mm Gape: Comprehensive Configuration Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['Config'], fontsize=9, rotation=0)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels inside bars
    for bar, mean, count in zip(bars, stats_df['Mean'], stats_df['Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{mean:.0f}\n(n={count})',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#f39c12', edgecolor='black', label='4.2mm Hook'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='4.5mm Hook')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    # Add horizontal line at mean
    overall_mean = df['Force_at_39mm_Gape_N'].mean()
    ax.axhline(y=overall_mean, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Overall Mean: {overall_mean:.0f} N')

    plt.tight_layout()
    output_file = output_dir / 'force_at_39mm_all_configurations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_side_by_side_comparison(df, output_dir):
    """
    Create side-by-side comparison of Force at 39mm and Force at Failure.

    Args:
        df: DataFrame with merged metrics and configuration data
        output_dir: Directory to save plots
    """
    # Create configuration label
    df['Config_Label'] = df.apply(create_config_label, axis=1)

    # Group by configuration and calculate statistics
    config_groups = df.groupby(['Hook Type', 'Hook Diameter', 'Mono Diameter', 'Crimp Type'])

    stats = []
    for config, group in config_groups:
        force_39_data = group['Force_at_39mm_Gape_N'].dropna()
        force_fail_data = group['Force_at_Failure_N'].dropna()

        if len(force_fail_data) > 0:
            config_label = create_config_label(group.iloc[0])
            stats.append({
                'Config': config_label,
                'Crimp Type': config[3],
                'Force_39_Mean': force_39_data.mean() if len(force_39_data) > 0 else 0,
                'Force_39_Std': force_39_data.std() if len(force_39_data) > 0 else 0,
                'Force_Fail_Mean': force_fail_data.mean(),
                'Force_Fail_Std': force_fail_data.std(),
                'Count': len(force_fail_data)
            })

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('Force_Fail_Mean', ascending=True)

    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    x_pos = np.arange(len(stats_df))

    # Plot 1: Force at 39mm Gape
    colors_39 = ['#3498db' if c == 'Jap C' else '#e74c3c' for c in stats_df['Crimp Type']]
    bars1 = ax1.bar(x_pos, stats_df['Force_39_Mean'], yerr=stats_df['Force_39_Std'],
                    capsize=5, alpha=0.85, color=colors_39, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Force at 39mm Gape (N)', fontsize=14, fontweight='bold')
    ax1.set_title('Force at 39mm Gape by Configuration', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(stats_df['Config'], fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, mean, count in zip(bars1, stats_df['Force_39_Mean'], stats_df['Count']):
        if mean > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{mean:.0f}\n(n={count})',
                    ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Plot 2: Force at Failure
    colors_fail = ['#3498db' if c == 'Jap C' else '#e74c3c' for c in stats_df['Crimp Type']]
    bars2 = ax2.bar(x_pos, stats_df['Force_Fail_Mean'], yerr=stats_df['Force_Fail_Std'],
                    capsize=5, alpha=0.85, color=colors_fail, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Configuration (Hook Type, Diameter / Mono Diameter, Crimp Type)',
                   fontsize=14, fontweight='bold')
    ax2.set_ylabel('Force at Failure (N)', fontsize=14, fontweight='bold')
    ax2.set_title('Force at Failure by Configuration', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stats_df['Config'], fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, mean, count in zip(bars2, stats_df['Force_Fail_Mean'], stats_df['Count']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                f'{mean:.0f}\n(n={count})',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Add legend to both
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Jap C Crimp'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Jap D Crimp')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11)
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'force_comparison_side_by_side.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 80)
    print("Generating Comprehensive Force Summary Plots")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_and_merge_data()
    print(f"  Loaded {len(df)} samples")

    # Create output directory
    output_dir = Path("results/batch/configuration_analysis/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comprehensive summary plots...")

    # Generate plots
    plot_comprehensive_force_summary(df, output_dir)
    plot_comprehensive_force_at_39mm_summary(df, output_dir)
    plot_side_by_side_comparison(df, output_dir)

    print("\n" + "=" * 80)
    print("Comprehensive Force Summary Plots Complete")
    print("=" * 80)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - force_at_failure_all_configurations.png")
    print("  - force_at_39mm_all_configurations.png")
    print("  - force_comparison_side_by_side.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
