#!/usr/bin/env python
"""
Analyze failure points across batches and configurations.

Generates:
- Failure point frequency by batch
- Failure point frequency by configuration parameters
- Relationship between failure point and force at failure
- Visualization of failure mode distributions

Usage (from project root):
    python scripts/batch/analyze_failure_points.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data_with_failure_points():
    """Load batch metrics merged with configuration and failure point data."""
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


def analyze_failure_points_by_batch(df, output_dir):
    """Analyze failure point distribution by batch."""
    # Count failure points by batch
    batch_failure_counts = df.groupby(['Batch', 'Failure Point']).size().unstack(fill_value=0)

    # Save to CSV
    output_file = output_dir / "failure_points_by_batch.csv"
    batch_failure_counts.to_csv(output_file)
    print(f"  ✓ Saved: {output_file.name}")

    # Calculate percentages
    batch_failure_pct = batch_failure_counts.div(batch_failure_counts.sum(axis=1), axis=0) * 100

    # Save percentages
    output_file_pct = output_dir / "failure_points_by_batch_percentage.csv"
    batch_failure_pct.to_csv(output_file_pct, float_format='%.1f')
    print(f"  ✓ Saved: {output_file_pct.name}")

    return batch_failure_counts, batch_failure_pct


def analyze_failure_points_by_config(df, output_dir):
    """Analyze failure point distribution by configuration parameters."""
    results = {}

    # By Hook Type
    hook_type_failure = df.groupby(['Hook Type', 'Failure Point']).size().unstack(fill_value=0)
    results['Hook Type'] = hook_type_failure

    # By Hook Diameter
    hook_diam_failure = df.groupby(['Hook Diameter', 'Failure Point']).size().unstack(fill_value=0)
    results['Hook Diameter'] = hook_diam_failure

    # By Mono Diameter
    mono_diam_failure = df.groupby(['Mono Diameter', 'Failure Point']).size().unstack(fill_value=0)
    results['Mono Diameter'] = mono_diam_failure

    # By Crimp Type
    crimp_failure = df.groupby(['Crimp Type', 'Failure Point']).size().unstack(fill_value=0)
    results['Crimp Type'] = crimp_failure

    # Save all to CSV
    for param, data in results.items():
        output_file = output_dir / f"failure_points_by_{param.lower().replace(' ', '_')}.csv"
        data.to_csv(output_file)
        print(f"  ✓ Saved: {output_file.name}")

    return results


def analyze_failure_point_force_relationship(df, output_dir):
    """Analyze relationship between failure point and force at failure."""
    # Group by failure point and calculate statistics
    failure_stats = df.groupby('Failure Point').agg({
        'Force_at_Failure_N': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'Gape_at_Failure_mm': ['mean', 'std'],
        'Time_to_Failure_s': ['mean', 'std']
    }).round(2)

    # Flatten column names
    failure_stats.columns = ['_'.join(col).strip() for col in failure_stats.columns.values]

    # Save to CSV
    output_file = output_dir / "failure_point_performance_statistics.csv"
    failure_stats.to_csv(output_file)
    print(f"  ✓ Saved: {output_file.name}")

    return failure_stats


def plot_failure_point_distribution_by_batch(df, output_dir):
    """Create stacked bar chart of failure points by batch."""
    # Count failure points by batch
    batch_failure_counts = df.groupby(['Batch', 'Failure Point']).size().unstack(fill_value=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Define colors for each failure point
    failure_colors = {
        'Mono above hook,below crimp': '#e74c3c',
        'Mono at looped knot on weight': '#3498db',
        'Hook Slipped': '#f39c12',
        'Mono above crimp at hook': '#9b59b6',
        'Mono below weight crimp': '#2ecc71',
        'Pulled through crimp at hook': '#e67e22',
        'Hook broke': '#95a5a6',
        'Pulled through crimp at weight': '#34495e',
        'Mono at hook ring': '#1abc9c',
        'Hook straightened off jig': '#d35400'
    }

    # Get columns in order of frequency
    col_order = batch_failure_counts.sum().sort_values(ascending=False).index

    # Create stacked bar chart
    batch_failure_counts[col_order].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=[failure_colors.get(col, '#7f8c8d') for col in col_order],
        edgecolor='black',
        linewidth=0.5,
        width=0.8
    )

    # Customize plot
    ax.set_xlabel('Batch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Failure Point Distribution by Batch', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(title='Failure Point', bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=10, title_fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'failure_points_by_batch_stacked.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_failure_point_force_comparison(df, output_dir):
    """Create box plot comparing force at failure by failure point."""
    # Filter to only common failure points (>5 occurrences)
    failure_counts = df['Failure Point'].value_counts()
    common_failures = failure_counts[failure_counts > 5].index

    df_filtered = df[df['Failure Point'].isin(common_failures)].copy()

    # Sort by median force
    failure_order = df_filtered.groupby('Failure Point')['Force_at_Failure_N'].median().sort_values().index

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create box plot
    positions = range(len(failure_order))
    data_to_plot = [df_filtered[df_filtered['Failure Point'] == fp]['Force_at_Failure_N'].dropna().values
                    for fp in failure_order]

    bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True,
                     showmeans=True, meanline=False,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markeredgecolor='black', markersize=8))

    # Customize plot
    ax.set_xlabel('Failure Point', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Force at Failure (N)', fontsize=14, fontweight='bold')
    ax.set_title('Force at Failure by Failure Point', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    # Split labels and rotate for better readability
    labels = []
    for fp in failure_order:
        # Split on comma and create multi-line label
        parts = fp.split(',')
        if len(parts) > 1:
            labels.append('\n'.join(parts))
        else:
            labels.append(fp)
    ax.set_xticklabels(labels, fontsize=9, rotation=0, ha='center')
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add sample counts
    for i, fp in enumerate(failure_order):
        count = len(df_filtered[df_filtered['Failure Point'] == fp])
        ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}',
                ha='center', fontsize=9, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color='red', label='Median'),
        Patch(color='green', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / 'force_at_failure_by_failure_point.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_failure_point_pie_charts(df, output_dir):
    """Create pie charts showing failure point distribution overall and by configuration."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Failure Point Distribution Analysis', fontsize=18, fontweight='bold', y=0.995)

    # Define colors
    failure_colors = {
        'Mono above hook,below crimp': '#e74c3c',
        'Mono at looped knot on weight': '#3498db',
        'Hook Slipped': '#f39c12',
        'Mono above crimp at hook': '#9b59b6',
        'Mono below weight crimp': '#2ecc71',
        'Pulled through crimp at hook': '#e67e22',
        'Hook broke': '#95a5a6'
    }

    # Overall distribution
    ax = axes[0, 0]
    failure_counts = df['Failure Point'].value_counts()
    colors = [failure_colors.get(fp, '#7f8c8d') for fp in failure_counts.index]
    ax.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 9})
    ax.set_title('Overall Distribution\n(All Samples)', fontsize=12, fontweight='bold')

    # By Hook Type
    ax = axes[0, 1]
    for hook_type in ['Round', 'Forged']:
        if hook_type == 'Round':
            ax_current = axes[0, 1]
            title = 'Round Hooks'
        else:
            ax_current = axes[0, 2]
            title = 'Forged Hooks'

        subset = df[df['Hook Type'] == hook_type]
        failure_counts = subset['Failure Point'].value_counts()
        colors = [failure_colors.get(fp, '#7f8c8d') for fp in failure_counts.index]
        ax_current.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 8})
        ax_current.set_title(f'{title}\n(n={len(subset)})', fontsize=12, fontweight='bold')

    # By Crimp Type
    for i, crimp_type in enumerate(['Jap C', 'Jap D']):
        ax = axes[1, i]
        subset = df[df['Crimp Type'] == crimp_type]
        failure_counts = subset['Failure Point'].value_counts()
        colors = [failure_colors.get(fp, '#7f8c8d') for fp in failure_counts.index]
        ax.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 8})
        ax.set_title(f'{crimp_type} Crimp\n(n={len(subset)})', fontsize=12, fontweight='bold')

    # By Mono Diameter
    ax = axes[1, 2]
    # Combine both mono diameters for comparison
    mono_21 = df[df['Mono Diameter'] == 2.1]['Failure Point'].value_counts()
    mono_23 = df[df['Mono Diameter'] == 2.3]['Failure Point'].value_counts()

    # Show 2.1mm
    failure_counts = mono_21
    colors = [failure_colors.get(fp, '#7f8c8d') for fp in failure_counts.index]
    ax.pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 8})
    ax.set_title(f'2.1mm Mono\n(n={df[df["Mono Diameter"]==2.1].shape[0]})',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'failure_points_pie_charts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def main():
    print("=" * 80)
    print("Failure Point Analysis")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_data_with_failure_points()
    print(f"  Loaded {len(df)} samples")

    # Create output directories
    output_dir = Path("results/batch/failure_point_analysis")
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Overall summary
    print("\n" + "=" * 80)
    print("Overall Failure Point Summary")
    print("=" * 80)
    failure_summary = df['Failure Point'].value_counts()
    print(failure_summary)

    # Analysis 1: By Batch
    print("\n" + "=" * 80)
    print("Analysis by Batch")
    print("=" * 80)
    batch_counts, batch_pct = analyze_failure_points_by_batch(df, output_dir)

    # Analysis 2: By Configuration
    print("\n" + "=" * 80)
    print("Analysis by Configuration Parameters")
    print("=" * 80)
    config_results = analyze_failure_points_by_config(df, output_dir)

    # Analysis 3: Force relationship
    print("\n" + "=" * 80)
    print("Failure Point vs Force Analysis")
    print("=" * 80)
    force_stats = analyze_failure_point_force_relationship(df, output_dir)
    print("\nForce at Failure by Failure Point:")
    print(force_stats[['Force_at_Failure_N_count', 'Force_at_Failure_N_mean',
                       'Force_at_Failure_N_std']].to_string())

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    plot_failure_point_distribution_by_batch(df, plots_dir)
    plot_failure_point_force_comparison(df, plots_dir)
    plot_failure_point_pie_charts(df, plots_dir)

    print("\n" + "=" * 80)
    print("Failure Point Analysis Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Plots saved to: {plots_dir}/")
    print("\nKey Files:")
    print("  - failure_points_by_batch.csv")
    print("  - failure_points_by_batch_percentage.csv")
    print("  - failure_point_performance_statistics.csv")
    print("  - Configuration-specific CSV files")
    print("\nPlots:")
    print("  - failure_points_by_batch_stacked.png")
    print("  - force_at_failure_by_failure_point.png")
    print("  - failure_points_pie_charts.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
