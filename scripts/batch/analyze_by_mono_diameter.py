#!/usr/bin/env python
"""
Generate separate statistical analyses for 2.1mm and 2.3mm mono diameter groups.

Creates complete analysis sets for each mono diameter including:
- Batch statistics
- Configuration comparisons
- Failure point analysis
- Visualizations

Usage (from project root):
    python scripts/batch/analyze_by_mono_diameter.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

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

    # Load master datasheet
    master_csv = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv")
    if not master_csv.exists():
        print(f"Error: {master_csv} not found.")
        sys.exit(1)

    master_df = pd.read_csv(master_csv)

    # Filter to only A-H batches
    batch_config = master_df[master_df['Test #'].str.match(r'^[A-H]\d+', na=False)].copy()
    batch_config = batch_config[['Test #', 'Hook Type', 'Hook Diameter', 'Hook Size',
                                   'Mono Diameter', 'Weight Size', 'Crimp Type', 'Crimp Style',
                                   'Failure Point', 'Secondary Failure Point']]
    batch_config.rename(columns={'Test #': 'Sample'}, inplace=True)

    # Merge
    merged_df = pd.merge(metrics_df, batch_config, on='Sample', how='left')

    # Add batch letter
    merged_df['Batch'] = merged_df['Sample'].str[0]

    return merged_df


def generate_batch_statistics(df, output_dir, mono_diam):
    """Generate batch-by-batch statistics."""
    # Define metric columns
    metric_cols = [
        'Force_at_39mm_Gape_N',
        'Force_at_Failure_N',
        'Time_to_39mm_Gape_s',
        'Time_to_Failure_s',
        'Gape_at_Failure_mm',
        'Delta_Gape_at_Failure_mm'
    ]

    # Calculate statistics by batch
    batch_stats = []
    for batch in sorted(df['Batch'].unique()):
        if batch is None:
            continue

        batch_data = df[df['Batch'] == batch]
        stats_row = {'Batch': batch, 'Sample_Count': len(batch_data)}

        for col in metric_cols:
            values = batch_data[col].dropna()
            if len(values) > 0:
                stats_row[f'{col}_Mean'] = values.mean()
                stats_row[f'{col}_Std'] = values.std()
                stats_row[f'{col}_Min'] = values.min()
                stats_row[f'{col}_Max'] = values.max()
                stats_row[f'{col}_Median'] = values.median()
            else:
                stats_row[f'{col}_Mean'] = None
                stats_row[f'{col}_Std'] = None
                stats_row[f'{col}_Min'] = None
                stats_row[f'{col}_Max'] = None
                stats_row[f'{col}_Median'] = None

        batch_stats.append(stats_row)

    batch_stats_df = pd.DataFrame(batch_stats)

    # Save
    output_file = output_dir / f"batch_statistics_{mono_diam}mm.csv"
    batch_stats_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"  ✓ Saved: {output_file.name}")

    return batch_stats_df


def generate_configuration_statistics(df, output_dir, mono_diam):
    """Generate statistics by configuration parameters."""
    metric_cols = [
        'Force_at_39mm_Gape_N',
        'Force_at_Failure_N',
        'Time_to_39mm_Gape_s',
        'Time_to_Failure_s',
        'Gape_at_Failure_mm',
        'Delta_Gape_at_Failure_mm'
    ]

    def calc_stats(group_df, group_cols):
        stats_list = []
        for group_vals, group_data in group_df.groupby(group_cols):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)

            stats_row = dict(zip(group_cols, group_vals))
            stats_row['Sample_Count'] = len(group_data)

            for metric in metric_cols:
                values = group_data[metric].dropna()
                if len(values) > 0:
                    stats_row[f'{metric}_Mean'] = values.mean()
                    stats_row[f'{metric}_Std'] = values.std()
                    stats_row[f'{metric}_Median'] = values.median()
                else:
                    stats_row[f'{metric}_Mean'] = None
                    stats_row[f'{metric}_Std'] = None
                    stats_row[f'{metric}_Median'] = None

            stats_list.append(stats_row)

        return pd.DataFrame(stats_list)

    # By Hook Type
    hook_type_stats = calc_stats(df, ['Hook Type'])
    hook_type_stats.to_csv(output_dir / f"by_hook_type_{mono_diam}mm.csv", index=False, float_format='%.4f')
    print(f"  ✓ Saved: by_hook_type_{mono_diam}mm.csv")

    # By Hook Diameter
    hook_diam_stats = calc_stats(df, ['Hook Diameter'])
    hook_diam_stats.to_csv(output_dir / f"by_hook_diameter_{mono_diam}mm.csv", index=False, float_format='%.4f')
    print(f"  ✓ Saved: by_hook_diameter_{mono_diam}mm.csv")

    # By Crimp Type
    crimp_stats = calc_stats(df, ['Crimp Type'])
    crimp_stats.to_csv(output_dir / f"by_crimp_type_{mono_diam}mm.csv", index=False, float_format='%.4f')
    print(f"  ✓ Saved: by_crimp_type_{mono_diam}mm.csv")

    # By Full Configuration
    full_config_stats = calc_stats(df, ['Hook Type', 'Hook Diameter', 'Crimp Type'])
    full_config_stats.to_csv(output_dir / f"by_full_configuration_{mono_diam}mm.csv", index=False, float_format='%.4f')
    print(f"  ✓ Saved: by_full_configuration_{mono_diam}mm.csv")

    return {
        'Hook Type': hook_type_stats,
        'Hook Diameter': hook_diam_stats,
        'Crimp Type': crimp_stats,
        'Full Config': full_config_stats
    }


def generate_failure_point_statistics(df, output_dir, mono_diam):
    """Generate failure point statistics."""
    # Overall distribution
    failure_counts = df['Failure Point'].value_counts()
    failure_counts.to_csv(output_dir / f"failure_point_counts_{mono_diam}mm.csv", header=['Count'])
    print(f"  ✓ Saved: failure_point_counts_{mono_diam}mm.csv")

    # By batch
    batch_failure = df.groupby(['Batch', 'Failure Point']).size().unstack(fill_value=0)
    batch_failure.to_csv(output_dir / f"failure_points_by_batch_{mono_diam}mm.csv")
    print(f"  ✓ Saved: failure_points_by_batch_{mono_diam}mm.csv")

    # Performance by failure point
    failure_stats = df.groupby('Failure Point').agg({
        'Force_at_Failure_N': ['count', 'mean', 'std', 'median'],
        'Gape_at_Failure_mm': ['mean', 'std'],
        'Time_to_Failure_s': ['mean', 'std']
    }).round(2)
    failure_stats.columns = ['_'.join(col).strip() for col in failure_stats.columns.values]
    failure_stats.to_csv(output_dir / f"failure_point_performance_{mono_diam}mm.csv")
    print(f"  ✓ Saved: failure_point_performance_{mono_diam}mm.csv")


def plot_batch_comparison(df, output_dir, mono_diam):
    """Create batch comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{mono_diam}mm Mono: Batch Comparison', fontsize=18, fontweight='bold')

    batches = sorted(df['Batch'].unique())

    # Force at 39mm
    ax = axes[0, 0]
    batch_means = []
    batch_stds = []
    for batch in batches:
        batch_data = df[df['Batch'] == batch]['Force_at_39mm_Gape_N'].dropna()
        batch_means.append(batch_data.mean() if len(batch_data) > 0 else 0)
        batch_stds.append(batch_data.std() if len(batch_data) > 0 else 0)

    x_pos = np.arange(len(batches))
    ax.bar(x_pos, batch_means, yerr=batch_stds, capsize=5, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
    ax.set_title('Force at 39mm Gape', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batches)
    ax.grid(axis='y', alpha=0.3)

    # Force at Failure
    ax = axes[0, 1]
    batch_means = []
    batch_stds = []
    for batch in batches:
        batch_data = df[df['Batch'] == batch]['Force_at_Failure_N'].dropna()
        batch_means.append(batch_data.mean() if len(batch_data) > 0 else 0)
        batch_stds.append(batch_data.std() if len(batch_data) > 0 else 0)

    ax.bar(x_pos, batch_means, yerr=batch_stds, capsize=5, alpha=0.8, edgecolor='black', color='coral')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Force (N)', fontsize=12, fontweight='bold')
    ax.set_title('Force at Failure', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batches)
    ax.grid(axis='y', alpha=0.3)

    # Time to Failure
    ax = axes[1, 0]
    batch_means = []
    batch_stds = []
    for batch in batches:
        batch_data = df[df['Batch'] == batch]['Time_to_Failure_s'].dropna()
        batch_means.append(batch_data.mean() if len(batch_data) > 0 else 0)
        batch_stds.append(batch_data.std() if len(batch_data) > 0 else 0)

    ax.bar(x_pos, batch_means, yerr=batch_stds, capsize=5, alpha=0.8, edgecolor='black', color='lightgreen')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Time to Failure', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batches)
    ax.grid(axis='y', alpha=0.3)

    # Gape at Failure
    ax = axes[1, 1]
    batch_means = []
    batch_stds = []
    for batch in batches:
        batch_data = df[df['Batch'] == batch]['Gape_at_Failure_mm'].dropna()
        batch_means.append(batch_data.mean() if len(batch_data) > 0 else 0)
        batch_stds.append(batch_data.std() if len(batch_data) > 0 else 0)

    ax.bar(x_pos, batch_means, yerr=batch_stds, capsize=5, alpha=0.8, edgecolor='black', color='mediumpurple')
    ax.set_xlabel('Batch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gape (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Gape at Failure', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(batches)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"batch_comparison_{mono_diam}mm.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_configuration_comparison(df, output_dir, mono_diam):
    """Create configuration comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{mono_diam}mm Mono: Configuration Parameter Comparison',
                 fontsize=18, fontweight='bold')

    params = [
        ('Hook Type', axes[0], ['#E63946', '#457B9D']),
        ('Hook Diameter', axes[1], ['#2E86AB', '#A23B72']),
    ]

    for param, ax, colors in params:
        param_values = sorted(df[param].dropna().unique())
        means = []
        stds = []
        counts = []

        for val in param_values:
            data = df[df[param] == val]['Force_at_Failure_N'].dropna()
            means.append(data.mean() if len(data) > 0 else 0)
            stds.append(data.std() if len(data) > 0 else 0)
            counts.append(len(data))

        x_pos = np.arange(len(param_values))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.85,
                      color=colors[:len(param_values)], edgecolor='black', linewidth=1.5)

        ax.set_xlabel(param, fontsize=13, fontweight='bold')
        ax.set_ylabel('Force at Failure (N)', fontsize=13, fontweight='bold')
        ax.set_title(f'By {param}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in param_values], fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Add labels
        for bar, mean, count in zip(bars, means, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{mean:.0f}\n(n={count})',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    output_file = output_dir / f"configuration_comparison_{mono_diam}mm.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_failure_point_distribution(df, output_dir, mono_diam):
    """Create failure point distribution plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{mono_diam}mm Mono: Failure Point Analysis', fontsize=18, fontweight='bold')

    # Pie chart
    failure_counts = df['Failure Point'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(failure_counts)))
    ax1.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 9})
    ax1.set_title(f'Failure Point Distribution\n(n={len(df)})', fontsize=13, fontweight='bold')

    # Force by failure point
    failure_points = df['Failure Point'].value_counts().index[:5]  # Top 5
    data_to_plot = []
    labels = []
    for fp in failure_points:
        data = df[df['Failure Point'] == fp]['Force_at_Failure_N'].dropna()
        if len(data) >= 3:  # Only include if enough samples
            data_to_plot.append(data.values)
            labels.append(f"{fp}\n(n={len(data)})")

    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax2.set_ylabel('Force at Failure (N)', fontsize=12, fontweight='bold')
        ax2.set_title('Force at Failure by Failure Point', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=10)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"failure_point_analysis_{mono_diam}mm.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 80)
    print("Statistical Analysis by Mono Diameter")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_and_merge_data()
    print(f"  Loaded {len(df)} total samples")

    # Process each mono diameter
    for mono_diam in [2.1, 2.3]:
        print("\n" + "=" * 80)
        print(f"Processing {mono_diam}mm Mono Diameter")
        print("=" * 80)

        # Filter data
        df_mono = df[df['Mono Diameter'] == mono_diam].copy()
        print(f"  Filtered to {len(df_mono)} samples")

        # Create output directory
        output_dir = Path(f"results/batch/mono_{mono_diam}mm_analysis")
        plots_dir = output_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate statistics
        print(f"\n  Generating batch statistics...")
        generate_batch_statistics(df_mono, output_dir, mono_diam)

        print(f"\n  Generating configuration statistics...")
        generate_configuration_statistics(df_mono, output_dir, mono_diam)

        print(f"\n  Generating failure point statistics...")
        generate_failure_point_statistics(df_mono, output_dir, mono_diam)

        # Generate plots
        print(f"\n  Generating visualizations...")
        plot_batch_comparison(df_mono, plots_dir, mono_diam)
        plot_configuration_comparison(df_mono, plots_dir, mono_diam)
        plot_failure_point_distribution(df_mono, plots_dir, mono_diam)

        print(f"\n  ✓ Completed {mono_diam}mm analysis")
        print(f"    Results saved to: {output_dir}/")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print("\nGenerated directories:")
    print("  - results/batch/mono_2.1mm_analysis/")
    print("  - results/batch/mono_2.3mm_analysis/")
    print("\nEach directory contains:")
    print("  - Batch statistics CSV")
    print("  - Configuration statistics CSVs")
    print("  - Failure point statistics CSVs")
    print("  - Visualization plots in plots/ subdirectory")
    print("=" * 80)


if __name__ == "__main__":
    main()
