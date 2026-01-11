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


def find_video_start(gape_data: Dict,
                     change_threshold_px: float = 3.0,
                     baseline_frames: int = 5,
                     min_consecutive: int = 3) -> float:
    """
    Find when the video tracking starts to deviate from the initial gape.

    Args:
        gape_data: Dictionary with 'time_seconds' and 'distance_pixels'
        change_threshold_px: Minimum absolute change (in pixels) from baseline
        baseline_frames: Number of initial frames used to compute baseline
        min_consecutive: Minimum consecutive frames above threshold

    Returns:
        start_time: Time in seconds when gape starts changing
    """
    gape_time = np.array(gape_data.get('time_seconds', []))
    gape_dist = np.array(gape_data.get('distance_pixels', []))

    if len(gape_dist) == 0:
        return 0.0

    baseline_count = min(baseline_frames, len(gape_dist))
    baseline = np.median(gape_dist[:baseline_count])
    delta = np.abs(gape_dist - baseline)

    if min_consecutive <= 1:
        indices = np.where(delta >= change_threshold_px)[0]
        return gape_time[indices[0]] if len(indices) > 0 else 0.0

    for idx in range(0, len(delta) - min_consecutive + 1):
        if np.all(delta[idx:idx + min_consecutive] >= change_threshold_px):
            return gape_time[idx]

    return 0.0


def auto_sync_video_to_mechanical(gape_data: Dict,
                                   shimadzu_df: pd.DataFrame,
                                   search_range: Tuple[float, float] = (0, 10),
                                   search_steps: int = 200,
                                   method: str = 'peak_alignment',
                                   force_threshold: float = 0.1,
                                   stroke_threshold: float = 0.01,
                                   video_change_threshold_px: float = 3.0,
                                   baseline_frames: int = 5,
                                   min_consecutive: int = 3,
                                   drop_smooth_window: int = 5,
                                   signature_force_weight: float = 0.7,
                                   signature_stroke_weight: float = 0.3,
                                   signature_smooth_window: int = 5) -> Tuple[float, float]:
    """
    Automatically synchronize video tracking data with mechanical test data.

    Strategy: For hook testing, align the point of maximum gape (just before failure)
    with the peak force in the mechanical test, as hook failure occurs at peak load.

    Args:
        gape_data: Dictionary with 'time_seconds' and 'distance_pixels'
        shimadzu_df: DataFrame with mechanical test data
        search_range: (min_offset, max_offset) in seconds to search for best alignment
        method: 'peak_alignment' (align max gape with peak force), 'start_alignment'
                (align start of movement), 'drop_alignment' (align sharp drop),
                'multi_signature' (align shape signatures), or 'correlation' (rate-based)
        force_threshold: Force threshold in N for start alignment
        stroke_threshold: Stroke threshold in mm for start alignment
        video_change_threshold_px: Pixel change threshold for video start alignment
        baseline_frames: Number of frames used to compute baseline
        min_consecutive: Minimum consecutive frames above threshold
        drop_smooth_window: Smoothing window for drop alignment (frames/samples)
        signature_force_weight: Weight for force rate signature in multi_signature
        signature_stroke_weight: Weight for stroke rate signature in multi_signature
        signature_smooth_window: Smoothing window for multi_signature (samples)

    Returns:
        best_offset: Time offset in seconds (add to video time to get mechanical time)
        correlation: Correlation coefficient at best offset (or None for peak_alignment)
    """
    gape_time = np.array(gape_data['time_seconds'])
    gape_dist = np.array(gape_data['distance_pixels'])

    if len(gape_dist) < 2:
        return 0.0, 0.0

    if method == 'start_alignment':
        mech_start_time = find_test_start(
            shimadzu_df,
            force_threshold=force_threshold,
            stroke_threshold=stroke_threshold
        )
        video_start_time = find_video_start(
            gape_data,
            change_threshold_px=video_change_threshold_px,
            baseline_frames=baseline_frames,
            min_consecutive=min_consecutive
        )
        best_offset = mech_start_time - video_start_time

        print("Start alignment sync:")
        print(f"  Video start at: {video_start_time:.2f}s (threshold={video_change_threshold_px:.2f} px)")
        print(f"  Mechanical start at: {mech_start_time:.2f}s (F>{force_threshold} N, stroke>{stroke_threshold} mm)")
        print(f"  Calculated offset: {best_offset:.2f}s")

        return best_offset, None

    if method == 'drop_alignment':
        def _smooth_series(series: np.ndarray, window: int) -> np.ndarray:
            if window <= 1 or len(series) < window:
                return series
            kernel = np.ones(window) / window
            return np.convolve(series, kernel, mode='same')

        def _find_drop_time(time: np.ndarray, series: np.ndarray, window: int) -> float:
            if len(series) < 2:
                return 0.0
            smoothed = _smooth_series(series, window)
            diff = np.diff(smoothed)
            if len(diff) == 0:
                return 0.0
            drop_idx = np.argmin(diff)
            return time[1:][drop_idx]

        force_time = shimadzu_df['Time'].values
        force_series = shimadzu_df['Force'].values
        drop_force_time = _find_drop_time(force_time, force_series, drop_smooth_window)

        gape_time = np.array(gape_data['time_seconds'])
        gape_series = np.array(gape_data['distance_pixels'])
        drop_gape_time = _find_drop_time(gape_time, gape_series, drop_smooth_window)

        best_offset = drop_force_time - drop_gape_time

        print("Drop alignment sync:")
        print(f"  Video drop at: {drop_gape_time:.2f}s")
        print(f"  Mechanical drop at: {drop_force_time:.2f}s")
        print(f"  Calculated offset: {best_offset:.2f}s")

        return best_offset, None

    if method == 'multi_signature':
        def _normalize(series: np.ndarray) -> np.ndarray:
            series = np.array(series, dtype=float)
            mean = np.nanmean(series)
            std = np.nanstd(series)
            if std == 0 or np.isnan(std):
                return np.zeros_like(series)
            return (series - mean) / std

        def _smooth_series(series: np.ndarray, window: int) -> np.ndarray:
            if window <= 1 or len(series) < window:
                return series
            kernel = np.ones(window) / window
            return np.convolve(series, kernel, mode='same')

        if len(gape_dist) < 2:
            return 0.0, 0.0

        gape_rate = np.gradient(gape_dist, gape_time)
        force_rate = np.gradient(shimadzu_df['Force'].values, shimadzu_df['Time'].values)
        stroke_rate = np.gradient(shimadzu_df['Stroke'].values, shimadzu_df['Time'].values)

        force_sig = _normalize(_smooth_series(force_rate, signature_smooth_window))
        stroke_sig = _normalize(_smooth_series(stroke_rate, signature_smooth_window))
        mech_signature = (signature_force_weight * force_sig) + (signature_stroke_weight * stroke_sig)
        mech_signature = _normalize(mech_signature)

        video_signature = _normalize(_smooth_series(gape_rate, signature_smooth_window))

        offsets = np.linspace(search_range[0], search_range[1], search_steps)
        correlations = []
        mech_time = shimadzu_df['Time'].values

        for offset in offsets:
            shifted_video_time = gape_time + offset
            video_interp = np.interp(
                mech_time,
                shifted_video_time,
                video_signature,
                left=np.nan, right=np.nan
            )
            valid_mask = np.isfinite(video_interp) & np.isfinite(mech_signature)
            if valid_mask.sum() > 10:
                corr = np.corrcoef(video_interp[valid_mask],
                                  mech_signature[valid_mask])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        best_idx = np.argmax(correlations)
        best_offset = offsets[best_idx]
        best_corr = correlations[best_idx]

        return best_offset, best_corr

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
        offsets = np.linspace(search_range[0], search_range[1], search_steps)
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

    initial_gape_px = None

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

    if pixels_per_mm:
        synced_df['Gape_Distance_mm'] = synced_df['Gape_Distance_px'] / pixels_per_mm
        if initial_gape_px is not None and initial_gape_mm is not None:
            synced_df['Gape_Distance_mm_corrected'] = (
                initial_gape_mm + (synced_df['Gape_Distance_px'] - initial_gape_px) / pixels_per_mm
            )
        else:
            synced_df['Gape_Distance_mm_corrected'] = np.nan
    else:
        synced_df['Gape_Distance_mm'] = np.nan
        synced_df['Gape_Distance_mm_corrected'] = np.nan

    return synced_df


def plot_synchronized_data(synced_df: pd.DataFrame,
                           output_path: Optional[str] = None,
                           pixels_per_mm: Optional[float] = None,
                           show_true_39mm: bool = False,
                           target_gape_mm: float = 39.0):
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

    # Gape distance (prefer corrected mm if available)
    use_corrected = (
        pixels_per_mm
        and 'Gape_Distance_mm_corrected' in synced_df.columns
        and synced_df['Gape_Distance_mm_corrected'].notna().any()
    )
    if use_corrected:
        valid_gape = synced_df['Gape_Distance_mm_corrected'].notna()
        gape_data = synced_df.loc[valid_gape, 'Gape_Distance_mm_corrected']
        gape_label = 'Gape Distance (mm, corrected)'
    else:
        valid_gape = synced_df['Gape_Distance_px'].notna()
        gape_data = synced_df.loc[valid_gape, 'Gape_Distance_px']
        gape_label = 'Gape Distance (px)'
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

    t_39 = None
    if show_true_39mm and use_corrected:
        target_mask = gape_data >= target_gape_mm
        if target_mask.any():
            t_39 = synced_df.loc[gape_data.index[target_mask], 'Time'].iloc[0]
            ax3a.axhline(target_gape_mm, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

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

    if t_39 is not None:
        for ax in axes:
            ax.axvline(t_39, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Synchronized plot saved to {output_path}")

    return fig
