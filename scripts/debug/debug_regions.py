#!/usr/bin/env python
"""
Debug script to visualize all detected green regions.
"""

import cv2
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker

def debug_regions():
    """Visualize all detected regions with labels."""

    # Load first frame
    cap = cv2.VideoCapture('data/Video/Batch_A1-A25/A1.mp4')
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    # Initialize tracker
    tracker = GreenPointTracker(hsv_lower=(35, 50, 50), hsv_upper=(85, 255, 255))

    # Find green regions
    mask, contours_info = tracker.find_green_regions(frame)

    print(f"Found {len(contours_info)} green regions")
    print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")

    # Create annotated image showing ALL regions
    result = frame.copy()

    print("\nAll regions (area > 100):")
    for i, info in enumerate(contours_info[:20]):  # Show top 20
        centroid = info['centroid']
        area = info['area']

        # Draw region
        cv2.drawContours(result, [info['contour']], -1, (0, 255, 255), 1)
        cv2.circle(result, centroid, 3, (0, 0, 255), -1)

        # Label with number and info
        label = f"{i+1}"
        cv2.putText(result, label, (centroid[0]+5, centroid[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        print(f"  Region {i+1}: Area={area:.0f}, Centroid={centroid}, BBox={info['bbox']}")

    # Save debug image
    cv2.imwrite('output/debug_all_regions.jpg', result)
    print("\nDebug image saved to output/debug_all_regions.jpg")
    print("Check this image to identify which region numbers correspond to:")
    print("  - Lighter green on shaft (should be upper region)")
    print("  - Lighter green on tip (should be at the hook point)")

if __name__ == "__main__":
    debug_regions()
