#!/usr/bin/env python
"""
Interactive tool to manually select the tip marker location.
"""

import cv2
import numpy as np

# Global variable to store the clicked point
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Selected point: ({x}, {y})")

def select_tip_location():
    """Let user click on the tip marker to identify it."""
    global clicked_point

    # Load first frame
    img = cv2.imread('first_frame.jpg')
    if img is None:
        print("Error: Could not load image")
        return

    display = img.copy()

    print("="*60)
    print("Tip Marker Selection Tool")
    print("="*60)
    print("Click on the CENTER of the lighter green paint on the TIP")
    print("(NOT the dark green loop - the lighter green at the hook point)")
    print("\nPress 'q' when done")
    print("="*60)

    cv2.namedWindow('Select Tip Marker')
    cv2.setMouseCallback('Select Tip Marker', mouse_callback)

    while True:
        if clicked_point:
            display = img.copy()
            cv2.circle(display, clicked_point, 10, (0, 0, 255), 2)
            cv2.putText(display, f"Tip: {clicked_point}",
                       (clicked_point[0] + 15, clicked_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Select Tip Marker', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if clicked_point:
        print("\n" + "="*60)
        print(f"Tip marker location: {clicked_point}")
        print("="*60)

        # Sample HSV values at that location
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Sample a 10x10 region around the clicked point
        x, y = clicked_point
        region = hsv[max(0, y-5):min(img.shape[0], y+5),
                     max(0, x-5):min(img.shape[1], x+5)]

        h_vals = region[:, :, 0].flatten()
        s_vals = region[:, :, 1].flatten()
        v_vals = region[:, :, 2].flatten()

        print(f"\nHSV values at tip marker:")
        print(f"  H: mean={np.mean(h_vals):.1f}, range=[{np.min(h_vals)}-{np.max(h_vals)}]")
        print(f"  S: mean={np.mean(s_vals):.1f}, range=[{np.min(s_vals)}-{np.max(s_vals)}]")
        print(f"  V: mean={np.mean(v_vals):.1f}, range=[{np.min(v_vals)}-{np.max(v_vals)}]")
        print("="*60)

        return clicked_point
    else:
        print("No point selected")
        return None

if __name__ == "__main__":
    tip_location = select_tip_location()

    if tip_location:
        print(f"\nUse this information to update the tip detection zone in tracker.py")
        print(f"Tip centroid should be around: {tip_location}")
