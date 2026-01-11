#!/usr/bin/env python
"""
Sample HSV values from a video at a point or a bounding box across frames.

Usage:
  python scripts/debug/sample_hsv_from_video.py <video_path> --point x,y
  python scripts/debug/sample_hsv_from_video.py <video_path> --bbox x1,y1,x2,y2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_point(value):
    try:
        x_str, y_str = value.split(",")
        return int(x_str), int(y_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid point: {value} (expected x,y)") from exc


def parse_bbox(value):
    try:
        x1_str, y1_str, x2_str, y2_str = value.split(",")
        x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid bbox: {value} (expected x1,y1,x2,y2)"
        ) from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("BBox must have x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def clamp_bbox(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sample_region(hsv, point=None, bbox=None, sample_radius=10):
    height, width = hsv.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = clamp_bbox(*bbox, width, height)
        if not (x2 > x1 and y2 > y1):
            return None
        return hsv[y1:y2, x1:x2]

    if point is None:
        return None
    x, y = point
    x1 = max(0, x - sample_radius)
    y1 = max(0, y - sample_radius)
    x2 = min(width, x + sample_radius)
    y2 = min(height, y + sample_radius)
    if x2 <= x1 or y2 <= y1:
        return None
    return hsv[y1:y2, x1:x2]


def collect_hsv_samples(
    cap,
    start_frame,
    num_frames,
    stride,
    point=None,
    bbox=None,
    sample_radius=10,
):
    h_vals = []
    s_vals = []
    v_vals = []

    for i in range(num_frames):
        frame_index = start_frame + (i * stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        region = sample_region(hsv, point=point, bbox=bbox, sample_radius=sample_radius)
        if region is None or region.size == 0:
            continue

        h_vals.append(region[:, :, 0].flatten())
        s_vals.append(region[:, :, 1].flatten())
        v_vals.append(region[:, :, 2].flatten())

    if not h_vals:
        return None, None, None

    h_vals = np.concatenate(h_vals)
    s_vals = np.concatenate(s_vals)
    v_vals = np.concatenate(v_vals)
    return h_vals, s_vals, v_vals


def summarize_channel(values):
    return {
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p5": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Sample HSV values from a video.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--point", type=parse_point, help="Sample point x,y")
    parser.add_argument("--bbox", type=parse_bbox, help="Sample bbox x1,y1,x2,y2")
    parser.add_argument(
        "--sample-radius",
        type=int,
        default=10,
        help="Radius in pixels around point (region is 2*radius square)",
    )
    parser.add_argument("--frame", type=int, default=0, help="Start frame index")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to sample",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride between samples",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=None,
        help="Start time in seconds (overrides --frame if set)",
    )
    parser.add_argument(
        "--emit-yaml",
        type=str,
        default=None,
        help="Sample name to emit a YAML snippet",
    )
    args = parser.parse_args()

    if args.point is None and args.bbox is None:
        parser.error("Provide --point or --bbox for sampling.")

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start_frame = args.frame
    if args.time is not None and fps > 0:
        start_frame = int(round(args.time * fps))

    h_vals, s_vals, v_vals = collect_hsv_samples(
        cap,
        start_frame,
        args.num_frames,
        args.stride,
        point=args.point,
        bbox=args.bbox,
        sample_radius=args.sample_radius,
    )
    cap.release()

    if h_vals is None:
        raise SystemExit("No HSV samples collected. Check point/bbox and frame range.")

    h_stats = summarize_channel(h_vals)
    s_stats = summarize_channel(s_vals)
    v_stats = summarize_channel(v_vals)

    h_lower = max(0, int(h_stats["p5"]))
    h_upper = min(179, int(h_stats["p95"]))
    s_lower = max(0, int(s_stats["p10"]))
    v_lower = max(0, int(v_stats["p10"]))

    print("=" * 70)
    print("HSV Sampling Summary")
    print("=" * 70)
    if args.bbox:
        print(f"Region: bbox={args.bbox}")
    else:
        print(f"Region: point={args.point} radius={args.sample_radius}px")
    print(f"Frames: start={start_frame} num={args.num_frames} stride={args.stride}")
    if fps > 0:
        print(f"FPS: {fps:.3f}")

    print("\nH channel:")
    print(f"  min={h_stats['min']} max={h_stats['max']} mean={h_stats['mean']:.2f} median={h_stats['median']:.2f} std={h_stats['std']:.2f}")
    print(f"  p5={h_stats['p5']:.2f} p95={h_stats['p95']:.2f}")
    print("\nS channel:")
    print(f"  min={s_stats['min']} max={s_stats['max']} mean={s_stats['mean']:.2f} median={s_stats['median']:.2f} std={s_stats['std']:.2f}")
    print(f"  p10={s_stats['p10']:.2f} p95={s_stats['p95']:.2f}")
    print("\nV channel:")
    print(f"  min={v_stats['min']} max={v_stats['max']} mean={v_stats['mean']:.2f} median={v_stats['median']:.2f} std={v_stats['std']:.2f}")
    print(f"  p10={v_stats['p10']:.2f} p95={v_stats['p95']:.2f}")

    print("\nSuggested HSV range:")
    print(f"  hsv_lower = ({h_lower}, {s_lower}, {v_lower})")
    print("  hsv_upper = ({} , 255, 255)".format(h_upper))
    print("\nCommand line:")
    print(f'  --hsv-lower "{h_lower},{s_lower},{v_lower}" --hsv-upper "{h_upper},255,255"')

    if args.emit_yaml:
        print("\nYAML snippet:")
        print("samples:")
        print(f"  {args.emit_yaml}:")
        print("    tracking:")
        print(f"      hsv_lower: [{h_lower}, {s_lower}, {v_lower}]")
        print(f"      hsv_upper: [{h_upper}, 255, 255]")

    print("=" * 70)


if __name__ == "__main__":
    main()
