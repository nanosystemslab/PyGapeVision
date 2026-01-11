#!/usr/bin/env python
"""
Analyze batch results grouped by configuration parameters.

Groups batches by:
- Hook Diameter (4.2mm vs 4.5mm)
- Mono Diameter (2.1mm vs 2.3mm)
- Crimp Type (Jap C vs Jap D)

Generates statistical comparisons to understand which configuration
parameters affect performance metrics.

Usage (from project root):
    python scripts/batch/analyze_by_configuration.py
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
    """
    Load batch metrics and merge with configuration data from master datasheet.

    Returns:
        DataFrame with metrics and configuration columns
    """
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


def calculate_statistics_by_group(df, group_cols, metric_cols):
    """
    Calculate statistics grouped by configuration parameters.

    Args:
        df: DataFrame with metrics and configuration columns
        group_cols: List of columns to group by
        metric_cols: List of metric columns to analyze

    Returns:
        DataFrame with statistics by group
    """
    stats_list = []

    for group_vals, group_data in df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        stats_row = dict(zip(group_cols, group_vals))
        stats_row['Sample_Count'] = len(group_data)

        for metric in metric_cols:
            values = group_data[metric].dropna()
            if len(values) > 0:
                stats_row[f'{metric}_Mean'] = values.mean()
                stats_row[f'{metric}_Std'] = values.std()
                stats_row[f'{metric}_Min'] = values.min()
                stats_row[f'{metric}_Max'] = values.max()
                stats_row[f'{metric}_Median'] = values.median()
                stats_row[f'{metric}_Count'] = len(values)
            else:
                stats_row[f'{metric}_Mean'] = None
                stats_row[f'{metric}_Std'] = None
                stats_row[f'{metric}_Min'] = None
                stats_row[f'{metric}_Max'] = None
                stats_row[f'{metric}_Median'] = None
                stats_row[f'{metric}_Count'] = 0

        stats_list.append(stats_row)

    return pd.DataFrame(stats_list)


def plot_comparison_by_parameter(df, param_col, metric_col, ylabel, title, output_file, colors=None):
    """
    Create bar plot comparing metric values across parameter levels.

    Args:
        df: DataFrame with data
        param_col: Configuration parameter column to group by
        metric_col: Metric to compare
        ylabel: Y-axis label
        title: Plot title
        output_file: Path to save plot
        colors: Optional list of colors for bars
    """
    # Calculate statistics
    stats = []
    param_values = sorted(df[param_col].dropna().unique())

    for param_val in param_values:
        param_data = df[df[param_col] == param_val][metric_col].dropna()
        if len(param_data) > 0:
            stats.append({
                'Parameter': str(param_val),
                'Mean': param_data.mean(),
                'Std': param_data.std(),
                'Count': len(param_data)
            })

    if not stats:
        print(f"  ⚠ No data for {param_col} vs {metric_col}")
        return

    stats_df = pd.DataFrame(stats)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(stats_df))
    if colors is None:
        colors = ['steelblue'] * len(stats_df)

    bars = ax.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std'],
                   capsize=8, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel(param_col, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['Parameter'], fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels inside bars at the top
    for bar, mean, std, count in zip(bars, stats_df['Mean'], stats_df['Std'], stats_df['Count']):
        height = bar.get_height()
        # Place text inside bar, near the top
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                f'{mean:.1f} ± {std:.1f}\nn={count}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_multi_parameter_comparison(df, metric_col, ylabel, title, output_file):
    """
    Create grouped bar plot comparing metric across multiple parameters.

    Args:
        df: DataFrame with data
        metric_col: Metric to compare
        ylabel: Y-axis label
        title: Plot title
        output_file: Path to save plot
    """
    # Group by Hook Diameter and Mono Diameter
    df['Config'] = df.apply(lambda row: f"Hook {row['Hook Diameter']}mm\nMono {row['Mono Diameter']}mm", axis=1)

    configs = sorted(df['Config'].dropna().unique())
    crimp_types = sorted(df['Crimp Type'].dropna().unique())

    # Calculate statistics
    stats = []
    for config in configs:
        for crimp in crimp_types:
            data = df[(df['Config'] == config) & (df['Crimp Type'] == crimp)][metric_col].dropna()
            if len(data) > 0:
                stats.append({
                    'Config': config,
                    'Crimp Type': crimp,
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Count': len(data)
                })

    if not stats:
        print(f"  ⚠ No data for multi-parameter comparison")
        return

    stats_df = pd.DataFrame(stats)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(configs))
    width = 0.35

    colors = {'Jap C': '#2E86AB', 'Jap D': '#A23B72'}

    for i, crimp in enumerate(crimp_types):
        crimp_data = stats_df[stats_df['Crimp Type'] == crimp]
        means = [crimp_data[crimp_data['Config'] == c]['Mean'].values[0]
                if len(crimp_data[crimp_data['Config'] == c]) > 0 else 0
                for c in configs]
        stds = [crimp_data[crimp_data['Config'] == c]['Std'].values[0]
               if len(crimp_data[crimp_data['Config'] == c]) > 0 else 0
               for c in configs]
        counts = [crimp_data[crimp_data['Config'] == c]['Count'].values[0]
                 if len(crimp_data[crimp_data['Config'] == c]) > 0 else 0
                 for c in configs]

        offset = width * (i - len(crimp_types)/2 + 0.5)
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=5,
                      label=crimp, alpha=0.8, color=colors.get(crimp, 'gray'),
                      edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar, mean, count in zip(bars, means, counts):
            if mean > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.0f}\n(n={count})',
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend(title='Crimp Type', fontsize=12, title_fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 80)
    print("Configuration-Based Statistical Analysis")
    print("=" * 80)

    # Load and merge data
    print("\nLoading data...")
    df = load_and_merge_data()
    print(f"  Loaded {len(df)} samples with configuration data")

    # Create output directories
    output_dir = Path("results/batch/configuration_analysis")
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Define metrics to analyze
    metric_cols = [
        'Force_at_39mm_Gape_N',
        'Force_at_Failure_N',
        'Time_to_39mm_Gape_s',
        'Time_to_Failure_s',
        'Gape_at_Failure_mm',
        'Delta_Gape_at_Failure_mm'
    ]

    # Analysis 1: By Hook Diameter
    print("\n" + "=" * 80)
    print("Analysis by Hook Diameter")
    print("=" * 80)
    hook_stats = calculate_statistics_by_group(df, ['Hook Diameter'], metric_cols)
    hook_stats_file = output_dir / "statistics_by_hook_diameter.csv"
    hook_stats.to_csv(hook_stats_file, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {hook_stats_file}")
    print("\nSummary:")
    print(hook_stats[['Hook Diameter', 'Sample_Count', 'Force_at_Failure_N_Mean',
                      'Force_at_39mm_Gape_N_Mean']].to_string(index=False))

    # Analysis 2: By Mono Diameter
    print("\n" + "=" * 80)
    print("Analysis by Mono Diameter")
    print("=" * 80)
    mono_stats = calculate_statistics_by_group(df, ['Mono Diameter'], metric_cols)
    mono_stats_file = output_dir / "statistics_by_mono_diameter.csv"
    mono_stats.to_csv(mono_stats_file, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {mono_stats_file}")
    print("\nSummary:")
    print(mono_stats[['Mono Diameter', 'Sample_Count', 'Force_at_Failure_N_Mean',
                      'Force_at_39mm_Gape_N_Mean']].to_string(index=False))

    # Analysis 3: By Hook Type
    print("\n" + "=" * 80)
    print("Analysis by Hook Type (Round vs Forged)")
    print("=" * 80)
    hook_type_stats = calculate_statistics_by_group(df, ['Hook Type'], metric_cols)
    hook_type_stats_file = output_dir / "statistics_by_hook_type.csv"
    hook_type_stats.to_csv(hook_type_stats_file, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {hook_type_stats_file}")
    print("\nSummary:")
    print(hook_type_stats[['Hook Type', 'Sample_Count', 'Force_at_Failure_N_Mean',
                           'Force_at_39mm_Gape_N_Mean']].to_string(index=False))

    # Analysis 4: By Crimp Type
    print("\n" + "=" * 80)
    print("Analysis by Crimp Type")
    print("=" * 80)
    crimp_stats = calculate_statistics_by_group(df, ['Crimp Type'], metric_cols)
    crimp_stats_file = output_dir / "statistics_by_crimp_type.csv"
    crimp_stats.to_csv(crimp_stats_file, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {crimp_stats_file}")
    print("\nSummary:")
    print(crimp_stats[['Crimp Type', 'Sample_Count', 'Force_at_Failure_N_Mean',
                       'Force_at_39mm_Gape_N_Mean']].to_string(index=False))

    # Analysis 5: By Combined Configuration
    print("\n" + "=" * 80)
    print("Analysis by Combined Configuration")
    print("=" * 80)
    config_stats = calculate_statistics_by_group(
        df, ['Hook Type', 'Hook Diameter', 'Mono Diameter', 'Crimp Type'], metric_cols
    )
    config_stats_file = output_dir / "statistics_by_full_configuration.csv"
    config_stats.to_csv(config_stats_file, index=False, float_format='%.4f')
    print(f"\n✓ Saved: {config_stats_file}")
    print("\nSummary:")
    print(config_stats[['Hook Type', 'Hook Diameter', 'Mono Diameter', 'Crimp Type',
                        'Sample_Count', 'Force_at_Failure_N_Mean']].to_string(index=False))

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Comparison Plots")
    print("=" * 80)

    # Hook Type comparisons
    plot_comparison_by_parameter(
        df, 'Hook Type', 'Force_at_Failure_N',
        'Force (N)', 'Force at Failure by Hook Type (Round vs Forged)',
        plots_dir / 'force_at_failure_by_hook_type.png',
        colors=['#E63946', '#457B9D']
    )

    plot_comparison_by_parameter(
        df, 'Hook Type', 'Force_at_39mm_Gape_N',
        'Force (N)', 'Force at 39mm Gape by Hook Type (Round vs Forged)',
        plots_dir / 'force_at_39mm_by_hook_type.png',
        colors=['#E63946', '#457B9D']
    )

    # Hook Diameter comparisons
    plot_comparison_by_parameter(
        df, 'Hook Diameter', 'Force_at_Failure_N',
        'Force (N)', 'Force at Failure by Hook Diameter',
        plots_dir / 'force_at_failure_by_hook_diameter.png',
        colors=['#2E86AB', '#A23B72']
    )

    plot_comparison_by_parameter(
        df, 'Hook Diameter', 'Force_at_39mm_Gape_N',
        'Force (N)', 'Force at 39mm Gape by Hook Diameter',
        plots_dir / 'force_at_39mm_by_hook_diameter.png',
        colors=['#2E86AB', '#A23B72']
    )

    # Mono Diameter comparisons
    plot_comparison_by_parameter(
        df, 'Mono Diameter', 'Force_at_Failure_N',
        'Force (N)', 'Force at Failure by Mono Diameter',
        plots_dir / 'force_at_failure_by_mono_diameter.png',
        colors=['#F18F01', '#C73E1D']
    )

    plot_comparison_by_parameter(
        df, 'Mono Diameter', 'Force_at_39mm_Gape_N',
        'Force (N)', 'Force at 39mm Gape by Mono Diameter',
        plots_dir / 'force_at_39mm_by_mono_diameter.png',
        colors=['#F18F01', '#C73E1D']
    )

    # Crimp Type comparisons
    plot_comparison_by_parameter(
        df, 'Crimp Type', 'Force_at_Failure_N',
        'Force (N)', 'Force at Failure by Crimp Type',
        plots_dir / 'force_at_failure_by_crimp_type.png',
        colors=['#6A994E', '#BC4B51']
    )

    plot_comparison_by_parameter(
        df, 'Crimp Type', 'Force_at_39mm_Gape_N',
        'Force (N)', 'Force at 39mm Gape by Crimp Type',
        plots_dir / 'force_at_39mm_by_crimp_type.png',
        colors=['#6A994E', '#BC4B51']
    )

    # Multi-parameter comparison
    plot_multi_parameter_comparison(
        df, 'Force_at_Failure_N',
        'Force (N)', 'Force at Failure: Hook & Mono Diameter by Crimp Type',
        plots_dir / 'force_at_failure_multi_parameter.png'
    )

    plot_multi_parameter_comparison(
        df, 'Force_at_39mm_Gape_N',
        'Force (N)', 'Force at 39mm Gape: Hook & Mono Diameter by Crimp Type',
        plots_dir / 'force_at_39mm_multi_parameter.png'
    )

    print("\n" + "=" * 80)
    print("Configuration Analysis Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Plots saved to: {plots_dir}/")
    print("\nKey Files:")
    print("  - statistics_by_hook_type.csv")
    print("  - statistics_by_hook_diameter.csv")
    print("  - statistics_by_mono_diameter.csv")
    print("  - statistics_by_crimp_type.csv")
    print("  - statistics_by_full_configuration.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
