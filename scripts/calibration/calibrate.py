#!/usr/bin/env python
"""
Calibration tool to determine pixels per mm using a ruler in the video.

Usage:
    python calibrate.py <video_path> <known_distance_mm>
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate pixel-to-mm conversion using a reference in the video'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file'
    )
    parser.add_argument(
        'known_distance_mm',
        type=float,
        help='Known distance in mm (e.g., 10 for 1cm on ruler)'
    )
    parser.add_argument(
        '--frame-number',
        type=int,
        default=0,
        help='Frame number to use for calibration (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration.json',
        help='Output file for calibration data (default: calibration.json)'
    )

    args = parser.parse_args()

    # Load video
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    # Skip to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {args.frame_number}")
        sys.exit(1)

    print("="*60)
    print("Calibration Tool")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Frame: {args.frame_number}")
    print(f"Known distance: {args.known_distance_mm} mm")
    print("\nInstructions:")
    print("1. Click on the start point of the known distance")
    print("2. Click on the end point of the known distance")
    print("3. Press any key to save and exit")
    print("="*60)

    points = []
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_frame
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            display_frame = frame.copy()

            # Draw points
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"P{i+1}", (pt[0]+10, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw line if both points selected
            if len(points) == 2:
                cv2.line(display_frame, points[0], points[1], (0, 255, 0), 2)

                # Calculate distance
                distance_pixels = np.sqrt((points[1][0] - points[0][0])**2 +
                                        (points[1][1] - points[0][1])**2)
                pixels_per_mm = distance_pixels / args.known_distance_mm

                # Display calibration info
                mid_x = (points[0][0] + points[1][0]) // 2
                mid_y = (points[0][1] + points[1][1]) // 2
                cv2.putText(display_frame, f"{distance_pixels:.1f} px = {args.known_distance_mm} mm",
                           (mid_x - 100, mid_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Calibration: {pixels_per_mm:.4f} px/mm",
                           (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow('Calibration', display_frame)

    cv2.imshow('Calibration', display_frame)
    cv2.setMouseCallback('Calibration', mouse_callback)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        distance_pixels = np.sqrt((points[1][0] - points[0][0])**2 +
                                (points[1][1] - points[0][1])**2)
        pixels_per_mm = distance_pixels / args.known_distance_mm

        # Save calibration
        import json
        calibration_data = {
            'video_path': str(video_path),
            'frame_number': args.frame_number,
            'known_distance_mm': args.known_distance_mm,
            'distance_pixels': float(distance_pixels),
            'pixels_per_mm': float(pixels_per_mm),
            'point1': points[0],
            'point2': points[1]
        }

        with open(args.output, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        # Save calibration image
        output_image = args.output.replace('.json', '.png')
        cv2.imwrite(output_image, display_frame)

        print("\n" + "="*60)
        print("Calibration Complete!")
        print("="*60)
        print(f"Distance in pixels: {distance_pixels:.2f} px")
        print(f"Calibration factor: {pixels_per_mm:.4f} px/mm")
        print(f"\nCalibration data saved to: {args.output}")
        print(f"Calibration image saved to: {output_image}")
        print("\nUse this value with analyze_hook.py:")
        print(f"  --pixels-per-mm {pixels_per_mm:.4f}")
        print("="*60)
    else:
        print("\nCalibration cancelled - need two points")
        sys.exit(1)


if __name__ == "__main__":
    main()
