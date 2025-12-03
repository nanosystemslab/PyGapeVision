"""
Utilities for working with the master datasheet.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_initial_gape_from_datasheet(
    sample_name: str,
    datasheet_path: Optional[str] = None
) -> Optional[float]:
    """
    Load the measured initial gape for a sample from the master datasheet.

    Args:
        sample_name: Sample identifier (e.g., "A1", "B15", "H25")
        datasheet_path: Path to master datasheet CSV. If None, uses default location.

    Returns:
        Initial gape in mm, or None if not found
    """
    if datasheet_path is None:
        # Default path
        datasheet_path = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv")
    else:
        datasheet_path = Path(datasheet_path)

    if not datasheet_path.exists():
        print(f"Warning: Master datasheet not found at {datasheet_path}")
        return None

    try:
        # Load datasheet
        df = pd.read_csv(datasheet_path)

        # Find row matching sample name
        row = df[df['Test #'] == sample_name]

        if len(row) == 0:
            print(f"Warning: Sample {sample_name} not found in datasheet")
            return None

        # Get initial gape value
        initial_gape = row.iloc[0]['Initial Gape']

        if pd.isna(initial_gape):
            print(f"Warning: Initial gape for {sample_name} is empty in datasheet")
            return None

        return float(initial_gape)

    except Exception as e:
        print(f"Warning: Error reading datasheet: {e}")
        return None


def load_all_initial_gapes(datasheet_path: Optional[str] = None) -> dict:
    """
    Load all initial gape measurements from the master datasheet.

    Args:
        datasheet_path: Path to master datasheet CSV. If None, uses default location.

    Returns:
        Dictionary mapping sample names to initial gape values (in mm)
    """
    if datasheet_path is None:
        datasheet_path = Path("data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv")
    else:
        datasheet_path = Path(datasheet_path)

    if not datasheet_path.exists():
        print(f"Warning: Master datasheet not found at {datasheet_path}")
        return {}

    try:
        df = pd.read_csv(datasheet_path)

        initial_gapes = {}
        for _, row in df.iterrows():
            sample_name = row['Test #']
            initial_gape = row['Initial Gape']

            if pd.notna(sample_name) and pd.notna(initial_gape):
                initial_gapes[str(sample_name)] = float(initial_gape)

        return initial_gapes

    except Exception as e:
        print(f"Warning: Error reading datasheet: {e}")
        return {}
