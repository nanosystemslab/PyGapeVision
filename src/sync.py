"""
Synchronization tools for aligning video tracking data with mechanical test data.
"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def load_shimadzu_csv(csv_path: str) -> pd.DataFrame:
    """
    Load Shimadzu CSV data.

    Args:
        csv_path: Path to Shimadzu CSV file

    Returns:
        DataFrame with Time, Force, and Stroke columns
    """
    # Read CSV, skipping first row (sample name), using row 2 as headers
    df = pd.read_csv(csv_path, skiprows=1)

    # Clean column names and convert to numeric
    df.columns = ['Time', 'Force', 'Stroke']
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Force'] = pd.to_numeric(df['Force'], errors='coerce')
    df['Stroke'] = pd.to_numeric(df['Stroke'], errors='coerce')

    # Drop any rows with NaN values
    df = df.dropna()

    return df


def find_test_start(shimadzu_df: pd.DataFrame,
                    force_threshold: float = 0.1,
                    stroke_threshold: float = 0.01) -> float:
    """
    Find when the mechanical test actually starts (when stroke begins moving significantly).

    Args:
        shimadzu_df: DataFrame with Force and Stroke data
        force_threshold: Force threshold in N to consider test started
        stroke_threshold: Stroke movement threshold in mm

    Returns:
        start_time: Time in seconds when test starts
    """
    # Find when stroke starts increasing consistently
    stroke_diff = shimadzu_df['Stroke'].diff()

    # Find first point where stroke is moving and force is positive
    mask = (shimadzu_df['Force'] > force_threshold) & (stroke_diff > stroke_threshold/10)

    if mask.any():
        start_idx = mask.idxmax()
        return shimadzu_df.loc[start_idx, 'Time']
    else:
        return 0.0


def auto_sync_video_to_mechanical(gape_data: Dict,
                                   shimadzu_df: pd.DataFrame,
                                   search_range: Tuple[float, float] = (0, 10),
                                   method: str = 'peak_alignment') -> Tuple[float, float]:
    """
    Automatically synchronize video tracking data with mechanical test data.

    Strategy: For hook testing, align the point of maximum gape (just before failure)
    with the peak force in the mechanical test, as hook failure occurs at peak load.

    Args:
        gape_data: Dictionary with 'time_seconds' and 'distance_pixels'
        shimadzu_df: DataFrame with mechanical test data
        search_range: (min_offset, max_offset) in seconds to search for best alignment
        method: 'peak_alignment' (align max gape with peak force) or 'correlation' (old method)

    Returns:
        best_offset: Time offset in seconds (add to video time to get mechanical time)
        correlation: Correlation coefficient at best offset (or None for peak_alignment)
    """
    gape_time = np.array(gape_data['time_seconds'])
    gape_dist = np.array(gape_data['distance_pixels'])

    if len(gape_dist) < 2:
        return 0.0, 0.0

    if method == 'peak_alignment':
        # Find point of maximum gape (just before failure)
        max_gape_idx = np.argmax(gape_dist)
        video_failure_time = gape_time[max_gape_idx]

        # Find peak force in mechanical test
        peak_force_idx = shimadzu_df['Force'].idxmax()
        mech_failure_time = shimadzu_df.loc[peak_force_idx, 'Time']

        # Calculate offset to align these two events
        best_offset = mech_failure_time - video_failure_time

        print(f"Peak alignment sync:")
        print(f"  Video max gape at: {video_failure_time:.2f}s (gape={gape_dist[max_gape_idx]:.1f} px)")
        print(f"  Mechanical peak force at: {mech_failure_time:.2f}s (F={shimadzu_df.loc[peak_force_idx, 'Force']:.1f} N)")
        print(f"  Calculated offset: {best_offset:.2f}s")

        return best_offset, None

    else:  # correlation method
        # Calculate rate of change of gape distance
        gape_rate = np.gradient(gape_dist, gape_time)

        # Calculate rate of change of stroke
        stroke_rate = shimadzu_df['Stroke'].diff() / shimadzu_df['Time'].diff()
        stroke_rate = stroke_rate.fillna(0)

        # Search for best offset
        offsets = np.linspace(search_range[0], search_range[1], 100)
        correlations = []

        for offset in offsets:
            # Shift video time by offset
            shifted_video_time = gape_time + offset

            # Interpolate gape rate to mechanical test timestamps
            gape_rate_interp = np.interp(
                shimadzu_df['Time'],
                shifted_video_time,
                gape_rate,
                left=0, right=0
            )

            # Calculate correlation
            valid_mask = (gape_rate_interp != 0) & (stroke_rate != 0)
            if valid_mask.sum() > 10:
                corr = np.corrcoef(gape_rate_interp[valid_mask],
                                  stroke_rate[valid_mask])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        # Find best offset
        best_idx = np.argmax(correlations)
        best_offset = offsets[best_idx]
        best_corr = correlations[best_idx]

        return best_offset, best_corr


def sync_data(gape_data: Dict,
              shimadzu_df: pd.DataFrame,
              time_offset: float,
              calculate_delta: bool = False,
              initial_gape_mm: Optional[float] = None,
              pixels_per_mm: Optional[float] = None) -> pd.DataFrame:
    """
    Create synchronized dataset combining gape and mechanical data.

    Args:
        gape_data: Dictionary with video tracking data
        shimadzu_df: DataFrame with mechanical test data
        time_offset: Time offset in seconds (add to video time to align with mechanical)
        calculate_delta: If True, calculate delta gape (change from initial position)
        initial_gape_mm: Known initial gape in mm (e.g., from manual measurement). If not provided,
                        uses the first tracked frame's gape.
        pixels_per_mm: Calibration factor (required if initial_gape_mm is provided)

    Returns:
        DataFrame with synchronized data on common timeline
    """
    # Shift video time
    gape_time = np.array(gape_data['time_seconds']) + time_offset
    gape_dist = np.array(gape_data['distance_pixels'])

    # Create combined dataframe on mechanical test timeline
    mech_time = shimadzu_df['Time'].values

    # Interpolate gape distance to mechanical timestamps
    gape_interp = np.interp(mech_time, gape_time, gape_dist, left=np.nan, right=np.nan)

    # Create synchronized dataframe
    synced_df = shimadzu_df.copy()
    synced_df['Gape_Distance_px'] = gape_interp
    synced_df['Video_Time_Offset'] = time_offset

    # Calculate delta gape if requested
    if calculate_delta and len(gape_dist) > 0:
        # Determine initial gape
        if initial_gape_mm is not None and pixels_per_mm is not None:
            # Use provided measured initial gape
            initial_gape_px = initial_gape_mm * pixels_per_mm
            print(f"Delta gape calculation enabled. Using measured initial gape: {initial_gape_mm:.2f} mm ({initial_gape_px:.2f} px)")
        else:
            # Use first tracked frame as initial gape
            initial_gape_px = gape_dist[0]
            initial_gape_mm = initial_gape_px / pixels_per_mm if pixels_per_mm else None
            print(f"Delta gape calculation enabled. Using first tracked frame as initial: {initial_gape_px:.2f} px")

        synced_df['Initial_Gape_px'] = initial_gape_px
        synced_df['Initial_Gape_mm'] = initial_gape_mm if initial_gape_mm is not None else np.nan
        synced_df['Delta_Gape_px'] = synced_df['Gape_Distance_px'] - initial_gape_px
    else:
        synced_df['Initial_Gape_px'] = np.nan
        synced_df['Initial_Gape_mm'] = np.nan
        synced_df['Delta_Gape_px'] = np.nan

    return synced_df


def plot_synchronized_data(synced_df: pd.DataFrame,
                           output_path: Optional[str] = None,
                           pixels_per_mm: Optional[float] = None):
    """
    Create comprehensive plot of synchronized gape and mechanical data.

    Args:
        synced_df: Synchronized DataFrame
        output_path: Path to save plot (optional)
        pixels_per_mm: Conversion factor for gape distance
    """
    # Check if delta gape is available
    has_delta = 'Delta_Gape_px' in synced_df.columns and synced_df['Delta_Gape_px'].notna().any()

    # Adjust number of subplots based on delta availability
    num_plots = 4 if has_delta else 3
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 10 if has_delta else 10), sharex=True)
    fig.suptitle('Synchronized Hook Gape and Mechanical Test Data',
                 fontsize=14, fontweight='bold')

    time = synced_df['Time']

    # Plot 1: Force vs Time
    axes[0].plot(time, synced_df['Force'], 'b-', linewidth=1.5, label='Force')
    axes[0].set_ylabel('Force (N)', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    axes[0].set_title('Tensile Force', fontsize=11)

    # Plot 2: Stroke vs Time
    axes[1].plot(time, synced_df['Stroke'], 'g-', linewidth=1.5, label='Stroke')
    axes[1].set_ylabel('Stroke (mm)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    axes[1].set_title('Crosshead Displacement', fontsize=11)

    # Plot 3: Gape Distance vs Time (with secondary axis for Force)
    ax3a = axes[2]

    # Gape distance
    valid_gape = synced_df['Gape_Distance_px'].notna()
    gape_label = 'Gape Distance (px)'
    gape_data = synced_df.loc[valid_gape, 'Gape_Distance_px']

    if pixels_per_mm:
        gape_data = gape_data / pixels_per_mm
        gape_label = 'Gape Distance (mm)'

    ax3a.plot(synced_df.loc[valid_gape, 'Time'], gape_data,
             'r-', linewidth=2, label='Gape Distance')
    ax3a.set_ylabel(gape_label, fontsize=11, fontweight='bold', color='r')
    ax3a.tick_params(axis='y', labelcolor='r')
    ax3a.grid(True, alpha=0.3)
    if not has_delta:
        ax3a.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax3a.set_title('Hook Gape vs Force', fontsize=11)

    # Force on secondary axis
    ax3b = ax3a.twinx()
    ax3b.plot(time, synced_df['Force'], 'b-', linewidth=1, alpha=0.6, label='Force')
    ax3b.set_ylabel('Force (N)', fontsize=11, fontweight='bold', color='b')
    ax3b.tick_params(axis='y', labelcolor='b')

    # Combined legend
    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 4: Delta Gape vs Time (if available)
    if has_delta:
        ax4a = axes[3]

        valid_delta = synced_df['Delta_Gape_px'].notna()
        delta_label = 'Delta Gape (px)'
        delta_data = synced_df.loc[valid_delta, 'Delta_Gape_px']

        if pixels_per_mm:
            delta_data = delta_data / pixels_per_mm
            delta_label = 'Delta Gape (mm)'

        ax4a.plot(synced_df.loc[valid_delta, 'Time'], delta_data,
                 'purple', linewidth=2, label='Delta Gape (from initial)')
        ax4a.set_ylabel(delta_label, fontsize=11, fontweight='bold', color='purple')
        ax4a.tick_params(axis='y', labelcolor='purple')
        ax4a.grid(True, alpha=0.3)
        ax4a.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax4a.set_title('Hook Gape Change (from initial position)', fontsize=11)

        # Force on secondary axis
        ax4b = ax4a.twinx()
        ax4b.plot(time, synced_df['Force'], 'b-', linewidth=1, alpha=0.6, label='Force')
        ax4b.set_ylabel('Force (N)', fontsize=11, fontweight='bold', color='b')
        ax4b.tick_params(axis='y', labelcolor='b')

        # Combined legend
        lines1, labels1 = ax4a.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Synchronized plot saved to {output_path}")

    return fig
