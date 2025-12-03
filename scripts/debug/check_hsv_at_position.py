#!/usr/bin/env python
"""
Check HSV values at specific positions in a video frame.

Usage:
    python scripts/debug/check_hsv_at_position.py <video_path> <x>,<y>
"""

import cv2
import sys
from pathlib import Path
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    if len(sys.argv) < 3:
        print("Usage: python check_hsv_at_position.py <video_path> <x>,<y>")
        print("Example: python check_hsv_at_position.py video.mp4 751,450")
        sys.exit(1)
    
    video_path = sys.argv[1]
    try:
        x, y = map(int, sys.argv[2].split(','))
    except:
        print(f"Error: Invalid position format: {sys.argv[2]}")
        print("Expected format: x,y (e.g., 751,450)")
        sys.exit(1)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        sys.exit(1)
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Sample region around the point
    sample_size = 20
    x1 = max(0, x - sample_size)
    y1 = max(0, y - sample_size)
    x2 = min(frame.shape[1], x + sample_size)
    y2 = min(frame.shape[0], y + sample_size)
    
    # Get BGR and HSV values at exact position
    bgr_value = frame[y, x]
    hsv_value = hsv[y, x]
    
    # Get average in region
    hsv_region = hsv[y1:y2, x1:x2]
    hsv_mean = np.mean(hsv_region, axis=(0, 1))
    hsv_std = np.std(hsv_region, axis=(0, 1))
    
    print("=" * 70)
    print(f"Color Analysis at Position ({x}, {y})")
    print("=" * 70)
    print(f"\nExact pixel:")
    print(f"  BGR: {bgr_value}")
    print(f"  HSV: {hsv_value}")
    print(f"\nRegion average ({sample_size}x{sample_size} pixels):")
    print(f"  H: {hsv_mean[0]:.1f} ± {hsv_std[0]:.1f}")
    print(f"  S: {hsv_mean[1]:.1f} ± {hsv_std[1]:.1f}")
    print(f"  V: {hsv_mean[2]:.1f} ± {hsv_std[2]:.1f}")
    print(f"\nCurrent detection range:")
    print(f"  Lower: (35, 100, 50)")
    print(f"  Upper: (55, 255, 255)")
    print(f"\nWould this be detected?")
    
    h, s, v = hsv_value
    in_range = (35 <= h <= 55) and (100 <= s <= 255) and (50 <= v <= 255)
    
    if in_range:
        print(f"  ✅ YES - HSV{tuple(hsv_value)} is within range")
    else:
        print(f"  ❌ NO - HSV{tuple(hsv_value)} is outside range")
        if h < 35 or h > 55:
            print(f"     Hue {h} is outside [35, 55]")
        if s < 100:
            print(f"     Saturation {s} is below 100 (too gray)")
        if v < 50:
            print(f"     Value {v} is below 50 (too dark)")
    
    print("\nSuggested range for this region:")
    h_min = max(0, int(hsv_mean[0] - hsv_std[0] * 2))
    h_max = min(179, int(hsv_mean[0] + hsv_std[0] * 2))
    s_min = max(0, int(hsv_mean[1] - hsv_std[1] * 2))
    v_min = max(0, int(hsv_mean[2] - hsv_std[2] * 2))
    print(f"  Lower: ({h_min}, {s_min}, {v_min})")
    print(f"  Upper: ({h_max}, 255, 255)")
    print("=" * 70)


if __name__ == "__main__":
    main()
