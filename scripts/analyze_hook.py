#!/usr/bin/env python
"""
Main script to analyze hook gape from video files.

Usage (from project root):
    python scripts/analyze_hook.py <video_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json
import cv2

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker, VideoAnalyzer
from src.visualization import plot_tracking_results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze hook gape distance from video files'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save output files (default: results)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Process every Nth frame (default: 1, process all frames)'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Do not create annotated output video'
    )
    parser.add_argument(
        '--pixels-per-mm',
        type=float,
        default=None,
        help='Calibration factor: pixels per mm (optional)'
    )
    parser.add_argument(
        '--hsv-lower',
        type=str,
        default='35,100,50',
        help='HSV lower bound for green detection (H,S,V) (default: 35,100,50)'
    )
    parser.add_argument(
        '--hsv-upper',
        type=str,
        default='55,255,255',
        help='HSV upper bound for green detection (H,S,V) (default: 55,255,255)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Override video frame rate (fps). If not specified, uses value from video metadata'
    )

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse HSV values
    hsv_lower = tuple(map(int, args.hsv_lower.split(',')))
    hsv_upper = tuple(map(int, args.hsv_upper.split(',')))

    print("="*60)
    print("PyGapeVision - Hook Gape Analysis")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"HSV lower bound: {hsv_lower}")
    print(f"HSV upper bound: {hsv_upper}")
    if args.fps:
        print(f"Frame rate (override): {args.fps} fps")
    if args.pixels_per_mm:
        print(f"Calibration: {args.pixels_per_mm:.4f} pixels/mm")
    print("="*60)

    # Initialize tracker
    tracker = GreenPointTracker(hsv_lower=hsv_lower, hsv_upper=hsv_upper)

    # Initialize analyzer
    analyzer = VideoAnalyzer(str(video_path), tracker)

    # Determine output paths
    video_name = video_path.stem
    output_video_path = None if args.no_video else str(output_dir / f"{video_name}_tracked.mp4")
    output_json_path = str(output_dir / f"{video_name}_results.json")
    output_plot_path = str(output_dir / f"{video_name}_analysis.png")

    # Process video
    print("\nProcessing video...")
    results = analyzer.process_video(
        output_video_path=output_video_path,
        frame_skip=args.frame_skip,
        fps_override=args.fps
    )

    # Save results
    print("\nSaving results...")
    analyzer.save_results(output_json_path)

    # Create plots
    print("\nGenerating analysis plots...")
    plot_tracking_results(
        results,
        output_path=output_plot_path,
        pixels_per_mm=args.pixels_per_mm
    )

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"Results JSON: {output_json_path}")
    print(f"Analysis plot: {output_plot_path}")
    if output_video_path:
        print(f"Tracked video: {output_video_path}")
    print("="*60)


if __name__ == "__main__":
    main()
