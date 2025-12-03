#!/usr/bin/env python
"""
Test script to verify edge validation is working correctly.
Processes a single frame and shows which regions pass/fail validation.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker

def main():
    # Load a test frame
    video_path = "data/Video/Batch_A1-A25/A1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return

    print(f"Frame shape: {frame.shape}")

    # Initialize tracker
    tracker = GreenPointTracker()

    # Find green regions
    mask, contours_info = tracker.find_green_regions(frame)

    print(f"\nFound {len(contours_info)} green regions")
    print("\nValidating tip candidates:")
    print("-" * 80)

    # Test each contour
    for i, contour_info in enumerate(contours_info[:10]):  # Test top 10
        cx, cy = contour_info['centroid']
        area = contour_info['area']

        # Test validation
        is_valid = tracker.validate_tip_edges(contour_info, frame=frame)

        print(f"Region {i+1}: centroid=({cx:4d}, {cy:4d}), area={area:6.0f}, "
              f"valid={is_valid}")

        # Draw on frame
        color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, red if not
        cv2.circle(frame, (cx, cy), 8, color, 2)
        cv2.putText(frame, f"{i+1}", (cx-10, cy-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated frame
    output_path = "results/debug/edge_validation_test.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\nAnnotated frame saved to: {output_path}")
    print("Green circles = valid tip (edges detected)")
    print("Red circles = invalid (no clear edges)")

if __name__ == "__main__":
    main()
