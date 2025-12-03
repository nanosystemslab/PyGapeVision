#!/usr/bin/env python
"""
Interactive tool to manually select shaft and tip positions from first frame.

Usage:
    python scripts/calibration/select_tracking_points.py <video_path>
    
Instructions:
    1. Click on the SHAFT marker (first click)
    2. Click on the TIP marker (second click)
    3. Press any key to confirm and exit
    
The script will output coordinates you can use with --init-shaft-pos and --init-tip-pos
"""

import cv2
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PointSelector:
    def __init__(self, image):
        self.image = image.copy()
        self.display_image = image.copy()
        self.shaft_pos = None
        self.tip_pos = None
        self.window_name = "Select Tracking Points"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.shaft_pos is None:
                self.shaft_pos = (x, y)
                # Draw shaft marker
                cv2.circle(self.display_image, (x, y), 10, (0, 0, 255), 2)
                cv2.putText(self.display_image, "SHAFT", (x - 30, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"Shaft position selected: ({x}, {y})")
            elif self.tip_pos is None:
                self.tip_pos = (x, y)
                # Draw tip marker
                cv2.circle(self.display_image, (x, y), 10, (255, 0, 0), 2)
                cv2.putText(self.display_image, "TIP", (x - 20, y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # Draw line between them
                cv2.line(self.display_image, self.shaft_pos, self.tip_pos, 
                        (0, 255, 255), 2)
                print(f"Tip position selected: ({x}, {y})")
                print("\nPress any key to confirm and exit...")
            
            cv2.imshow(self.window_name, self.display_image)
    
    def select_points(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Add instructions to image
        instructions = [
            "1. Click on SHAFT marker",
            "2. Click on TIP marker",
            "3. Press any key to finish"
        ]
        y_offset = 30
        for i, text in enumerate(instructions):
            cv2.putText(self.display_image, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.display_image)
        
        print("=" * 70)
        print("Manual Tracking Point Selection")
        print("=" * 70)
        print("Instructions:")
        print("  1. Click on the SHAFT marker (green paint on hook shaft)")
        print("  2. Click on the TIP marker (green paint on hook tip)")
        print("  3. Press any key when done")
        print("=" * 70)
        print()
        
        # Wait for user input
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return self.shaft_pos, self.tip_pos


def main():
    if len(sys.argv) < 2:
        print("Usage: python select_tracking_points.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read first frame")
        sys.exit(1)
    
    # Select points
    selector = PointSelector(frame)
    shaft_pos, tip_pos = selector.select_points()
    
    if shaft_pos and tip_pos:
        print("\n" + "=" * 70)
        print("Selection Complete!")
        print("=" * 70)
        print(f"Shaft position: {shaft_pos}")
        print(f"Tip position:   {tip_pos}")
        print()
        print("Use these positions with:")
        print(f"  --init-shaft-pos {shaft_pos[0]},{shaft_pos[1]}")
        print(f"  --init-tip-pos {tip_pos[0]},{tip_pos[1]}")
        print("=" * 70)
    else:
        print("\nError: Both shaft and tip positions must be selected")
        sys.exit(1)


if __name__ == "__main__":
    main()
