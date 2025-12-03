#!/usr/bin/env python
"""
Interactive HSV tuning tool to find the best color range for tracking.
"""

import cv2
import numpy as np

def nothing(x):
    pass

def tune_hsv(image_path):
    """Interactive HSV tuner with trackbars."""

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create window
    cv2.namedWindow('HSV Tuner')
    cv2.namedWindow('Original')

    # Create trackbars for HSV ranges
    cv2.createTrackbar('H Lower', 'HSV Tuner', 35, 180, nothing)
    cv2.createTrackbar('H Upper', 'HSV Tuner', 85, 180, nothing)
    cv2.createTrackbar('S Lower', 'HSV Tuner', 50, 255, nothing)
    cv2.createTrackbar('S Upper', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'HSV Tuner', 50, 255, nothing)
    cv2.createTrackbar('V Upper', 'HSV Tuner', 255, 255, nothing)

    print("="*60)
    print("HSV Color Tuner")
    print("="*60)
    print("Adjust the trackbars to isolate the lighter green paint")
    print("on the shaft and tip (exclude the darker green loop)")
    print("\nControls:")
    print("  - Adjust trackbars to tune HSV range")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current settings")
    print("="*60)

    while True:
        # Get current trackbar positions
        h_lower = cv2.getTrackbarPos('H Lower', 'HSV Tuner')
        h_upper = cv2.getTrackbarPos('H Upper', 'HSV Tuner')
        s_lower = cv2.getTrackbarPos('S Lower', 'HSV Tuner')
        s_upper = cv2.getTrackbarPos('S Upper', 'HSV Tuner')
        v_lower = cv2.getTrackbarPos('V Lower', 'HSV Tuner')
        v_upper = cv2.getTrackbarPos('V Upper', 'HSV Tuner')

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create mask
        lower = np.array([h_lower, s_lower, v_lower])
        upper = np.array([h_upper, s_upper, v_upper])
        mask = cv2.inRange(hsv, lower, upper)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours and mark them
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = img.copy()

        # Draw contours and number them by area
        contour_info = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    contour_info.append((area, cx, cy, cnt))

        # Sort by area
        contour_info.sort(reverse=True)

        # Draw top regions
        for i, (area, cx, cy, cnt) in enumerate(contour_info[:10]):
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result, f"{i+1}: {area:.0f}px", (cx+10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add text with current values
        text = f"H:[{h_lower}-{h_upper}] S:[{s_lower}-{s_upper}] V:[{v_lower}-{v_upper}]"
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, f"Regions found: {len(contour_info)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show images
        cv2.imshow('HSV Tuner', result)
        cv2.imshow('Original', mask)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n" + "="*60)
            print("Current HSV Settings:")
            print("="*60)
            print(f"hsv_lower = ({h_lower}, {s_lower}, {v_lower})")
            print(f"hsv_upper = ({h_upper}, {s_upper}, {v_upper})")
            print("\nCommand line arguments:")
            print(f'--hsv-lower "{h_lower},{s_lower},{v_lower}" --hsv-upper "{h_upper},{s_upper},{v_upper}"')
            print("="*60)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tune_hsv('first_frame.jpg')
