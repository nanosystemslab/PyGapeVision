#!/usr/bin/env python
"""
Populate the master datasheet with results from batch analysis.

Reads the master datasheet with initial gape measurements and fills in:
- Time to 39mm Gape
- Force at 39mm Gape
- Time to failure
- Gape at Failure

Usage (from project root):
    python scripts/batch/populate_master_datasheet.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import load_calibration


def find_metrics_from_results(csv_path, pixels_per_mm, target_total_gape_mm=39.0):
    """
    Extract metrics from a synchronized results CSV using delta gape.

    Args:
        csv_path: Path to synchronized CSV file
        pixels_per_mm: Calibration factor
        target_total_gape_mm: Target total gape in mm (e.g., 39mm)

    Returns:
        Dictionary with metrics (time_to_39mm, force_at_39mm, time_to_failure, gape_at_failure)
    """
    try:
        df = pd.read_csv(csv_path)

        # Check if Delta_Gape_px column exists
        if 'Delta_Gape_px' in df.columns and 'Initial_Gape_px' in df.columns:
            # Get initial gape in mm
            initial_gape_mm = df['Initial_Gape_px'].iloc[0] / pixels_per_mm

            # Calculate target delta gape: delta = total - initial
            target_delta_mm = target_total_gape_mm - initial_gape_mm

            # Convert delta gape from pixels to mm
            df['Delta_Gape_mm'] = df['Delta_Gape_px'] / pixels_per_mm

            # Find first row where delta gape >= target
            matches_39mm = df[df['Delta_Gape_mm'] >= target_delta_mm]
            time_to_39mm = matches_39mm.iloc[0]['Time'] if len(matches_39mm) > 0 else None
            force_at_39mm = matches_39mm.iloc[0]['Force'] if len(matches_39mm) > 0 else None
        else:
            # Fallback to absolute gape if delta not available
            df['Gape_mm'] = df['Gape_Distance_px'] / pixels_per_mm
            matches_39mm = df[df['Gape_mm'] >= target_total_gape_mm]
            time_to_39mm = matches_39mm.iloc[0]['Time'] if len(matches_39mm) > 0 else None
            force_at_39mm = matches_39mm.iloc[0]['Force'] if len(matches_39mm) > 0 else None

        # Failure metrics (at max force)
        max_force_idx = df['Force'].idxmax()
        time_to_failure = df.loc[max_force_idx, 'Time']
        df['Gape_mm'] = df['Gape_Distance_px'] / pixels_per_mm
        gape_at_failure = df.loc[max_force_idx, 'Gape_mm']

        return {
            'time_to_39mm': time_to_39mm,
            'force_at_39mm': force_at_39mm,
            'time_to_failure': time_to_failure,
            'gape_at_failure': gape_at_failure
        }

    except Exception as e:
        print(f"    Error extracting metrics: {e}")
        return {
            'time_to_39mm': None,
            'force_at_39mm': None,
            'time_to_failure': None,
            'gape_at_failure': None
        }


def main():
    print("=" * 70)
    print("Populating Master Datasheet")
    print("=" * 70)

    # Paths
    master_csv = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv")
    batch_results_dir = Path("results/batch")
    output_csv = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests_FILLED.csv")

    if not master_csv.exists():
        print(f"Error: Master datasheet not found at {master_csv}")
        sys.exit(1)

    # Load master datasheet
    print(f"\nLoading master datasheet: {master_csv}")
    master_df = pd.read_csv(master_csv)
    print(f"  Loaded {len(master_df)} samples")

    # Get calibration
    pixels_per_mm = load_calibration()
    print(f"  Using calibration: {pixels_per_mm:.3f} pixels/mm")

    # Process each sample
    print(f"\nProcessing samples...")

    updated_count = 0
    missing_count = 0

    for idx, row in master_df.iterrows():
        test_id = row['Test #']

        # Skip if not a standard test (e.g., Test 1-10 vs A1-H25)
        if not isinstance(test_id, str):
            continue

        # Determine batch and sample paths
        # Format: A1 -> Batch_A/A1, B15 -> Batch_B/B15
        if len(test_id) >= 2:
            batch_letter = test_id[0]

            # Look for results in batch directory
            batch_dir = batch_results_dir / f"Batch_{batch_letter}"
            sample_dir = batch_dir / test_id
            csv_file = sample_dir / f"{test_id}_synchronized.csv"

            if csv_file.exists():
                print(f"  ✓ {test_id}", end="")

                # Extract metrics
                metrics = find_metrics_from_results(csv_file, pixels_per_mm)

                # Update dataframe
                master_df.at[idx, 'Time to 39mm Gape'] = metrics['time_to_39mm']
                master_df.at[idx, 'Force at 39mm Gape'] = metrics['force_at_39mm']
                master_df.at[idx, 'Time to failure'] = metrics['time_to_failure']
                master_df.at[idx, 'Gape at Failure'] = metrics['gape_at_failure']

                updated_count += 1
                print(f" - Updated")
            else:
                print(f"  ⚠ {test_id} - No results found at {csv_file}")
                missing_count += 1

    # Save updated datasheet
    master_df.to_csv(output_csv, index=False)

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Samples updated: {updated_count}")
    print(f"  Samples missing: {missing_count}")
    print(f"\nUpdated datasheet saved to: {output_csv}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
