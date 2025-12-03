#!/usr/bin/env python
"""
Analyze HSV values in different regions to find the right thresholds.
"""

import cv2
import numpy as np

def analyze_regions():
    """Sample HSV values from different green regions."""
    img = cv2.imread('first_frame.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("="*60)
    print("HSV Color Analysis")
    print("="*60)

    # Sample regions - approximate coordinates from visual inspection
    # Region 1: Lighter green on shaft (top area)
    shaft_region = hsv[320:380, 690:760]
    # Region 2: Lighter green on tip (bottom)
    tip_region = hsv[560:620, 690:760]
    # Region 3: Darker green on loop (left side)
    loop_region = hsv[420:480, 400:470]

    regions = [
        ("Lighter Green (Shaft)", shaft_region),
        ("Lighter Green (Tip)", tip_region),
        ("Darker Green (Loop)", loop_region)
    ]

    for name, region in regions:
        h_vals = region[:, :, 0].flatten()
        s_vals = region[:, :, 1].flatten()
        v_vals = region[:, :, 2].flatten()

        print(f"\n{name}:")
        print(f"  H: mean={np.mean(h_vals):.1f}, min={np.min(h_vals)}, max={np.max(h_vals)}, median={np.median(h_vals):.1f}")
        print(f"  S: mean={np.mean(s_vals):.1f}, min={np.min(s_vals)}, max={np.max(s_vals)}, median={np.median(s_vals):.1f}")
        print(f"  V: mean={np.mean(v_vals):.1f}, min={np.min(v_vals)}, max={np.max(v_vals)}, median={np.median(v_vals):.1f}")

    # Calculate recommended thresholds
    # Combine lighter green samples
    lighter_h = np.concatenate([regions[0][1][:, :, 0].flatten(), regions[1][1][:, :, 0].flatten()])
    lighter_s = np.concatenate([regions[0][1][:, :, 1].flatten(), regions[1][1][:, :, 1].flatten()])
    lighter_v = np.concatenate([regions[0][1][:, :, 2].flatten(), regions[1][1][:, :, 2].flatten()])

    # Use percentiles for robust thresholds
    h_lower = int(np.percentile(lighter_h, 5))
    h_upper = int(np.percentile(lighter_h, 95))
    s_lower = int(np.percentile(lighter_s, 10))
    s_upper = 255
    v_lower = int(np.percentile(lighter_v, 10))
    v_upper = 255

    print("\n" + "="*60)
    print("Recommended HSV Range (for lighter green only):")
    print("="*60)
    print(f"hsv_lower = ({h_lower}, {s_lower}, {v_lower})")
    print(f"hsv_upper = ({h_upper}, {s_upper}, {v_upper})")
    print("\nCommand line arguments:")
    print(f'--hsv-lower "{h_lower},{s_lower},{v_lower}" --hsv-upper "{h_upper},{s_upper},{v_upper}"')
    print("="*60)

    # Test the thresholds
    lower = np.array([h_lower, s_lower, v_lower])
    upper = np.array([h_upper, s_upper, v_upper])
    mask = cv2.inRange(hsv, lower, upper)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\nFound {len(contours)} regions with recommended thresholds")

    # Save test mask
    cv2.imwrite('output/hsv_analysis_mask.jpg', mask)
    print("Test mask saved to output/hsv_analysis_mask.jpg")

    # Create annotated image
    result = img.copy()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 100:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result, f"{i+1}", (cx+10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite('output/hsv_analysis_result.jpg', result)
    print("Annotated result saved to output/hsv_analysis_result.jpg")

if __name__ == "__main__":
    analyze_regions()
