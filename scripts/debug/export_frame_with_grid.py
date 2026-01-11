#!/usr/bin/env python
"""
Export a video frame with a coordinate grid overlay for manual pixel lookup.

Usage:
  python scripts/debug/export_frame_with_grid.py <video_path> --output <path>
"""

import argparse
from pathlib import Path

import cv2


def draw_grid(frame, step, color=(0, 255, 255), text_color=(255, 255, 255)):
    height, width = frame.shape[:2]
    overlay = frame.copy()

    for x in range(0, width, step):
        cv2.line(overlay, (x, 0), (x, height - 1), color, 1)
        if x % (step * 2) == 0:
            cv2.putText(
                overlay,
                str(x),
                (x + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    for y in range(0, height, step):
        cv2.line(overlay, (0, y), (width - 1, y), color, 1)
        if y % (step * 2) == 0:
            cv2.putText(
                overlay,
                str(y),
                (5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Export a frame with grid overlay.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (png/jpg)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to export (default: 0)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Time in seconds to export (overrides --frame if set)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Grid spacing in pixels",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_index = args.frame
    if args.time is not None and fps > 0:
        frame_index = int(round(args.time * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise SystemExit("Could not read frame. Check --frame/--time.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = draw_grid(frame, args.step)
    cv2.imwrite(str(output_path), grid)
    print(f"Saved grid overlay to {output_path}")


if __name__ == "__main__":
    main()
