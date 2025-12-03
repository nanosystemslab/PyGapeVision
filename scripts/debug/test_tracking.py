#!/usr/bin/env python
"""
Quick test script to verify tracking works on a few frames.
"""

import cv2
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker

def test_tracking():
    """Test tracking on the first frame."""
    print("Testing green point tracker...")

    # Load first frame
    cap = cv2.VideoCapture('data/Video/Batch_A1-A25/A1.mp4')
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame")
        return False

    # Initialize tracker
    tracker = GreenPointTracker(hsv_lower=(35, 50, 50), hsv_upper=(85, 255, 255))

    # Find green regions
    mask, contours_info = tracker.find_green_regions(frame)

    print(f"Found {len(contours_info)} green regions")
    print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")

    # Print top 20 regions for debugging
    print("\nTop 20 regions by area:")
    for i, info in enumerate(contours_info[:20]):
        cx, cy = info['centroid']
        area = info['area']
        in_shaft_zone = (380 < cy < 520 and 620 < cx < 850)
        in_tip_zone = (650 < cy < 800 and 800 < cx < 1000)
        dist_to_target = ((cx - 902)**2 + (cy - 707)**2)**0.5 if in_tip_zone else None
        dist_str = f"{dist_to_target:.1f}" if dist_to_target is not None else "N/A"
        print(f"  {i+1}. Area: {area:.0f}, Centroid: ({cx}, {cy}), Shaft: {in_shaft_zone}, Tip: {in_tip_zone}, Dist to (902,707): {dist_str}")

    # Identify shaft and tip
    shaft_info, tip_info = tracker.identify_shaft_and_tip(contours_info, frame_width=frame.shape[1], frame_height=frame.shape[0])

    if shaft_info and tip_info:
        print(f"Shaft centroid: {shaft_info['centroid']}")
        print(f"Tip centroid: {tip_info['centroid']}")

        distance = tracker.calculate_distance(shaft_info['centroid'], tip_info['centroid'])
        print(f"Distance: {distance:.2f} pixels")

        # Create annotated frame
        annotated = tracker.annotate_frame(frame, shaft_info, tip_info, distance)

        # Save test outputs
        cv2.imwrite('output/test_mask.jpg', mask)
        cv2.imwrite('output/test_annotated.jpg', annotated)

        print("\nTest successful!")
        print("Outputs saved:")
        print("  - output/test_mask.jpg (green detection mask)")
        print("  - output/test_annotated.jpg (annotated frame)")
        return True
    else:
        print("Error: Could not identify both shaft and tip")
        # Save mask for debugging
        cv2.imwrite('output/test_mask_debug.jpg', mask)
        print("Debug mask saved to output/test_mask_debug.jpg")
        return False


if __name__ == "__main__":
    test_tracking()
