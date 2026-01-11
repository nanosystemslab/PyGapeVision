#!/usr/bin/env python
"""
Generate statistical analysis for each batch (A-H) from batch metrics.

Calculates mean, standard deviation, min, max, and count for each metric
by batch group.

Usage (from project root):
    python scripts/batch/analyze_batch_statistics.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_batch_letter(sample_name):
    """Extract batch letter from sample name (e.g., 'A1' -> 'A')"""
    if isinstance(sample_name, str) and len(sample_name) > 0:
        return sample_name[0]
    return None


def calculate_batch_statistics(df):
    """
    Calculate statistics for each batch.

    Args:
        df: DataFrame with Sample column and metrics columns

    Returns:
        DataFrame with statistics by batch
    """
    # Extract batch letter
    df['Batch'] = df['Sample'].apply(extract_batch_letter)

    # Define metric columns (exclude Sample and Batch)
    metric_columns = [col for col in df.columns if col not in ['Sample', 'Batch']]

    # Calculate statistics for each batch
    batch_stats = []

    for batch in sorted(df['Batch'].unique()):
        if batch is None:
            continue

        batch_data = df[df['Batch'] == batch]

        stats_row = {'Batch': batch, 'Sample_Count': len(batch_data)}

        for col in metric_columns:
            # Get non-null values
            values = batch_data[col].dropna()
            valid_count = len(values)

            if valid_count > 0:
                stats_row[f'{col}_Mean'] = values.mean()
                stats_row[f'{col}_Std'] = values.std()
                stats_row[f'{col}_Min'] = values.min()
                stats_row[f'{col}_Max'] = values.max()
                stats_row[f'{col}_Count'] = valid_count
            else:
                stats_row[f'{col}_Mean'] = None
                stats_row[f'{col}_Std'] = None
                stats_row[f'{col}_Min'] = None
                stats_row[f'{col}_Max'] = None
                stats_row[f'{col}_Count'] = 0

        batch_stats.append(stats_row)

    return pd.DataFrame(batch_stats)


def create_summary_by_metric(df):
    """
    Create a more readable summary organized by metric.

    Args:
        df: DataFrame with Sample column and metrics columns

    Returns:
        Dictionary of DataFrames, one per metric
    """
    # Extract batch letter
    df['Batch'] = df['Sample'].apply(extract_batch_letter)

    # Define metric columns (exclude Sample and Batch)
    metric_columns = [col for col in df.columns if col not in ['Sample', 'Batch']]

    metric_summaries = {}

    for metric in metric_columns:
        summary_data = []

        for batch in sorted(df['Batch'].unique()):
            if batch is None:
                continue

            batch_data = df[df['Batch'] == batch][metric].dropna()

            if len(batch_data) > 0:
                summary_data.append({
                    'Batch': batch,
                    'Count': len(batch_data),
                    'Mean': batch_data.mean(),
                    'Std': batch_data.std(),
                    'Min': batch_data.min(),
                    'Max': batch_data.max(),
                    'Median': batch_data.median()
                })
            else:
                summary_data.append({
                    'Batch': batch,
                    'Count': 0,
                    'Mean': None,
                    'Std': None,
                    'Min': None,
                    'Max': None,
                    'Median': None
                })

        metric_summaries[metric] = pd.DataFrame(summary_data)

    return metric_summaries


def main():
    print("=" * 80)
    print("Batch Statistical Analysis")
    print("=" * 80)

    # Define paths
    metrics_csv = Path("results/batch/batch_metrics_summary.csv")
    output_stats_csv = Path("results/batch/batch_statistics_summary.csv")
    output_dir = Path("results/batch/statistics_by_metric")

    if not metrics_csv.exists():
        print(f"\nError: Metrics file not found at {metrics_csv}")
        print("Please run collect_batch_metrics.py first.")
        sys.exit(1)

    # Load batch metrics
    print(f"\nLoading batch metrics from: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    print(f"  Loaded {len(df)} samples")

    # Calculate overall batch statistics
    print("\nCalculating batch statistics...")
    batch_stats_df = calculate_batch_statistics(df)

    # Save overall statistics
    batch_stats_df.to_csv(output_stats_csv, index=False, float_format='%.4f')
    print(f"  Saved comprehensive statistics to: {output_stats_csv}")

    # Create metric-specific summaries
    print("\nGenerating metric-specific summaries...")
    metric_summaries = create_summary_by_metric(df)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each metric summary
    for metric_name, metric_df in metric_summaries.items():
        output_file = output_dir / f"{metric_name}_by_batch.csv"
        metric_df.to_csv(output_file, index=False, float_format='%.4f')
        print(f"  ✓ {metric_name}")

    print(f"\nMetric-specific summaries saved to: {output_dir}/")

    # Display summary
    print(f"\n{'=' * 80}")
    print("Batch Summary Statistics")
    print(f"{'=' * 80}\n")

    # Show sample counts per batch
    print("Sample Counts by Batch:")
    print(batch_stats_df[['Batch', 'Sample_Count']].to_string(index=False))

    # Display key metrics for each batch
    print("\n" + "=" * 80)
    print("Key Metrics by Batch")
    print("=" * 80)

    for metric_name, metric_df in metric_summaries.items():
        print(f"\n{metric_name}:")
        print("-" * 80)
        display_df = metric_df.copy()
        # Format for better display
        for col in ['Mean', 'Std', 'Min', 'Max', 'Median']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        print(display_df.to_string(index=False))

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
