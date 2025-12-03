#!/usr/bin/env python
"""
Example script showing how to use the calibration system.

Run from project root:
    python scripts/calibration/example_calibration_usage.py
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationManager, load_calibration


def main():
    print("=== PyGapeVision Calibration System Example ===\n")
    
    # Initialize calibration manager
    cal_manager = CalibrationManager()
    
    # Example 1: Get default calibration
    print("1. Getting default calibration:")
    default_cal = cal_manager.get_pixels_per_mm()
    print(f"   Default: {default_cal:.3f} pixels/mm\n")
    
    # Example 2: Set batch-specific calibration
    print("2. Setting batch-specific calibration for Batch_B:")
    cal_manager.set_calibration(
        pixels_per_mm=12.250,
        batch_name="Batch_B",
        notes="Calibrated with different camera distance"
    )
    print(f"   Batch B: {cal_manager.get_pixels_per_mm(batch_name='Batch_B'):.3f} pixels/mm\n")
    
    # Example 3: Set video-specific calibration
    print("3. Setting video-specific calibration for C5.mp4:")
    cal_manager.set_calibration(
        pixels_per_mm=11.980,
        video_name="C5.mp4",
        notes="Camera repositioned for this sample"
    )
    print(f"   C5.mp4: {cal_manager.get_pixels_per_mm(video_name='C5.mp4'):.3f} pixels/mm\n")
    
    # Example 4: Priority demonstration
    print("4. Calibration priority demonstration:")
    print(f"   A1.mp4 (uses default): {cal_manager.get_pixels_per_mm(video_name='A1.mp4'):.3f} pixels/mm")
    print(f"   B1.mp4 (uses Batch_B): {cal_manager.get_pixels_per_mm(video_name='B1.mp4', batch_name='Batch_B'):.3f} pixels/mm")
    print(f"   C5.mp4 (uses video-specific): {cal_manager.get_pixels_per_mm(video_name='C5.mp4', batch_name='Batch_C'):.3f} pixels/mm")
    
    # Example 5: Using convenience function
    print("\n5. Using convenience function:")
    cal_value = load_calibration(video_name="A1.mp4", batch_name="Batch_A")
    print(f"   load_calibration('A1.mp4', 'Batch_A'): {cal_value:.3f} pixels/mm")
    
    print("\n=== Calibration data saved to results/calibration/calibration.json ===")


if __name__ == "__main__":
    main()
