"""
Visualization tools for tracking results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_tracking_results(results: Dict, output_path: Optional[str] = None,
                         pixels_per_mm: Optional[float] = None):
    """
    Create comprehensive plots of tracking results.

    Args:
        results: Dictionary containing tracking data
        output_path: Path to save the plot (optional)
        pixels_per_mm: Conversion factor from pixels to mm (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hook Gape Tracking Analysis', fontsize=16, fontweight='bold')

    time = np.array(results['time_seconds'])
    distance = np.array(results['distance_pixels'])

    # Convert to mm if calibration is provided
    distance_label = 'Distance (pixels)'
    if pixels_per_mm:
        distance = distance / pixels_per_mm
        distance_label = 'Gape Distance (mm)'

    # Plot 1: Distance vs Time
    axes[0, 0].plot(time, distance, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_ylabel(distance_label, fontsize=12)
    axes[0, 0].set_title('Gape Distance vs Time', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distance change (rate)
    if len(distance) > 1:
        distance_change = np.diff(distance) / np.diff(time)
        axes[0, 1].plot(time[1:], distance_change, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=12)
        ylabel = 'Rate of Change (pixels/s)' if not pixels_per_mm else 'Rate of Change (mm/s)'
        axes[0, 1].set_ylabel(ylabel, fontsize=12)
        axes[0, 1].set_title('Rate of Gape Change', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Plot 3: Trajectory of both points
    shaft_x = np.array(results['shaft_x'])
    shaft_y = np.array(results['shaft_y'])
    tip_x = np.array(results['tip_x'])
    tip_y = np.array(results['tip_y'])

    # Plot trajectories
    axes[1, 0].plot(shaft_x, shaft_y, 'r-', alpha=0.6, linewidth=1, label='Shaft trajectory')
    axes[1, 0].plot(tip_x, tip_y, 'b-', alpha=0.6, linewidth=1, label='Tip trajectory')

    # Mark start and end points
    axes[1, 0].plot(shaft_x[0], shaft_y[0], 'ro', markersize=10, label='Shaft start')
    axes[1, 0].plot(tip_x[0], tip_y[0], 'bo', markersize=10, label='Tip start')
    axes[1, 0].plot(shaft_x[-1], shaft_y[-1], 'r^', markersize=10, label='Shaft end')
    axes[1, 0].plot(tip_x[-1], tip_y[-1], 'b^', markersize=10, label='Tip end')

    axes[1, 0].set_xlabel('X Position (pixels)', fontsize=12)
    axes[1, 0].set_ylabel('Y Position (pixels)', fontsize=12)
    axes[1, 0].set_title('Point Trajectories', fontsize=13, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].invert_yaxis()  # Invert Y-axis to match image coordinates
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')

    # Plot 4: Statistics summary
    axes[1, 1].axis('off')

    stats_text = f"""
    TRACKING STATISTICS
    {'='*40}

    Total frames tracked: {len(time)}
    Duration: {time[-1]:.2f} seconds

    Initial gape: {distance[0]:.2f} {'px' if not pixels_per_mm else 'mm'}
    Final gape: {distance[-1]:.2f} {'px' if not pixels_per_mm else 'mm'}
    Maximum gape: {np.max(distance):.2f} {'px' if not pixels_per_mm else 'mm'}
    Minimum gape: {np.min(distance):.2f} {'px' if not pixels_per_mm else 'mm'}

    Total change: {distance[-1] - distance[0]:.2f} {'px' if not pixels_per_mm else 'mm'}
    Percent change: {((distance[-1] - distance[0]) / distance[0] * 100):.1f}%

    Mean gape: {np.mean(distance):.2f} {'px' if not pixels_per_mm else 'mm'}
    Std deviation: {np.std(distance):.2f} {'px' if not pixels_per_mm else 'mm'}
    """

    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    return fig


def create_calibration_helper(frame: np.ndarray, known_length_mm: float,
                              output_path: Optional[str] = None) -> float:
    """
    Interactive calibration helper to determine pixels per mm.

    Args:
        frame: Image frame containing a ruler or known reference
        known_length_mm: Known length in mm (e.g., from ruler)
        output_path: Path to save calibration image (optional)

    Returns:
        pixels_per_mm: Conversion factor
    """
    print("Calibration helper - Click two points on a known distance")
    print(f"Known distance: {known_length_mm} mm")

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Calibration', frame)

            if len(points) == 2:
                # Draw line between points
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                distance_pixels = np.sqrt((points[1][0] - points[0][0])**2 +
                                        (points[1][1] - points[0][1])**2)

                pixels_per_mm = distance_pixels / known_length_mm
                cv2.putText(frame, f"{distance_pixels:.1f} px = {known_length_mm} mm",
                           (points[0][0], points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Calibration: {pixels_per_mm:.2f} px/mm",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Calibration', frame)

    import cv2
    cv2.imshow('Calibration', frame)
    cv2.setMouseCallback('Calibration', mouse_callback)

    print("Click two points on the known distance. Press 'q' when done.")
    while len(points) < 2:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        distance_pixels = np.sqrt((points[1][0] - points[0][0])**2 +
                                (points[1][1] - points[0][1])**2)
        pixels_per_mm = distance_pixels / known_length_mm

        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"Calibration image saved to {output_path}")

        print(f"Calibration factor: {pixels_per_mm:.4f} pixels/mm")
        return pixels_per_mm
    else:
        print("Calibration failed - need two points")
        return None


def plot_force_vs_gape(synced_df: pd.DataFrame,
                       output_path: Optional[str] = None,
                       pixels_per_mm: Optional[float] = None):
    """
    Create force vs gape distance plot from synchronized data.

    Args:
        synced_df: Synchronized DataFrame with Force and Gape_Distance_px columns
        output_path: Path to save the plot (optional)
        pixels_per_mm: Conversion factor from pixels to mm (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out NaN gape values
    valid_data = synced_df[synced_df['Gape_Distance_px'].notna()].copy()

    if len(valid_data) == 0:
        print("Warning: No valid synchronized gape data to plot")
        return None

    gape = valid_data['Gape_Distance_px'].values
    force = valid_data['Force'].values

    # Convert to mm if calibration is provided
    gape_label = 'Gape Distance (pixels)'
    if pixels_per_mm:
        gape = gape / pixels_per_mm
        gape_label = 'Gape Distance (mm)'

    # Plot gape vs force (gape on y-axis, force on x-axis)
    ax.plot(force, gape, 'b-', linewidth=2, alpha=0.7)

    # Mark start and end points
    ax.plot(force[0], gape[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(force[-1], gape[-1], 'ro', markersize=12, label='End', zorder=5)

    # Find and mark peak force
    max_force_idx = np.argmax(force)
    max_force = force[max_force_idx]
    gape_at_max_force = gape[max_force_idx]
    ax.plot(max_force, gape_at_max_force, 'r*', markersize=20,
            label=f'Peak Force: {max_force:.1f} N', zorder=5)

    # Add annotations for peak
    ax.annotate(f'{max_force:.1f} N\n@ {gape_at_max_force:.2f} {"mm" if pixels_per_mm else "px"}',
                xy=(max_force, gape_at_max_force),
                xytext=(20, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

    # Find and mark force at 39mm gape (if calibrated)
    if pixels_per_mm:
        target_gape_mm = 39.0
        # Find index where gape is closest to 39mm
        gape_diff = np.abs(gape - target_gape_mm)
        target_idx = np.argmin(gape_diff)

        # Only mark if we found a point reasonably close to 39mm (within 5mm)
        if gape_diff[target_idx] < 5.0:
            force_at_39mm = force[target_idx]
            actual_gape = gape[target_idx]

            ax.plot(force_at_39mm, actual_gape, 'mo', markersize=15,
                    label=f'Force at 39mm: {force_at_39mm:.1f} N', zorder=5)

            ax.annotate(f'{force_at_39mm:.1f} N\n@ {actual_gape:.2f} mm',
                        xy=(force_at_39mm, actual_gape),
                        xytext=(-60, -40), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='magenta', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

    ax.set_xlabel('Force (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel(gape_label, fontsize=14, fontweight='bold')
    ax.set_title('Hook Gape vs Force', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    # Add statistics text box
    stats_text = f"""Statistics:
Peak Force: {max_force:.2f} N
Gape at Peak: {gape_at_max_force:.2f} {"mm" if pixels_per_mm else "px"}
Initial Gape: {gape[0]:.2f} {"mm" if pixels_per_mm else "px"}
Final Gape: {gape[-1]:.2f} {"mm" if pixels_per_mm else "px"}
Gape Change: {gape[-1] - gape[0]:.2f} {"mm" if pixels_per_mm else "px"}"""

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Gape vs Force plot saved to {output_path}")

    return fig


def plot_stroke_vs_force(synced_df: pd.DataFrame,
                         output_path: Optional[str] = None):
    """
    Create stroke vs force plot from synchronized data.

    Args:
        synced_df: Synchronized DataFrame with Force and Stroke columns
        output_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out any invalid data
    valid_data = synced_df.dropna(subset=['Force', 'Stroke']).copy()

    if len(valid_data) == 0:
        print("Warning: No valid synchronized data to plot")
        return None

    stroke = valid_data['Stroke'].values
    force = valid_data['Force'].values

    # Plot force vs stroke (force on y-axis, stroke on x-axis)
    ax.plot(stroke, force, 'g-', linewidth=2, alpha=0.7)

    # Mark start and end points
    ax.plot(stroke[0], force[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(stroke[-1], force[-1], 'ro', markersize=12, label='End', zorder=5)

    # Find and mark peak force
    max_force_idx = np.argmax(force)
    max_force = force[max_force_idx]
    stroke_at_max_force = stroke[max_force_idx]
    ax.plot(stroke_at_max_force, max_force, 'r*', markersize=20,
            label=f'Peak Force: {max_force:.1f} N', zorder=5)

    # Add annotations
    ax.annotate(f'{max_force:.1f} N\n@ {stroke_at_max_force:.2f} mm',
                xy=(stroke_at_max_force, max_force),
                xytext=(20, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

    ax.set_xlabel('Stroke (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Force (N)', fontsize=14, fontweight='bold')
    ax.set_title('Force vs Crosshead Displacement (Stroke)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    # Add statistics text box
    stats_text = f"""Statistics:
Peak Force: {max_force:.2f} N
Stroke at Peak: {stroke_at_max_force:.2f} mm
Initial Stroke: {stroke[0]:.2f} mm
Final Stroke: {stroke[-1]:.2f} mm
Total Displacement: {stroke[-1] - stroke[0]:.2f} mm"""

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Stroke vs Force plot saved to {output_path}")

    return fig
