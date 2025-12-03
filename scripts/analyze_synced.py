#!/usr/bin/env python
"""
Synchronized analysis combining video tracking with mechanical test data.

Usage (from project root):
    python scripts/analyze_synced.py <video_path> <csv_path> [options]
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker, VideoAnalyzer
from src.visualization import plot_tracking_results, plot_force_vs_gape, plot_stroke_vs_force
from src.sync import (load_shimadzu_csv, find_test_start, auto_sync_video_to_mechanical,
                     sync_data, plot_synchronized_data)
from src.datasheet import load_initial_gape_from_datasheet


def main():
    parser = argparse.ArgumentParser(
        description='Synchronized analysis of hook gape tracking and mechanical test data'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to Shimadzu CSV file'
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
        default=5,
        help='Process every Nth frame (default: 5)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=100,
        help='Video frame rate in fps (default: 100)'
    )
    parser.add_argument(
        '--time-offset',
        type=float,
        default=None,
        help='Manual time offset in seconds (video time + offset = mechanical time). If not provided, auto-sync is used.'
    )
    parser.add_argument(
        '--pixels-per-mm',
        type=float,
        default=None,
        help='Calibration factor: pixels per mm (optional)'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Do not create annotated output video'
    )
    parser.add_argument(
        '--rotate',
        action='store_true',
        help='Rotate output video 90 degrees clockwise'
    )
    parser.add_argument(
        '--init-shaft-pos',
        type=str,
        default=None,
        help='Initial shaft position as "x,y" (e.g., "724,460") for manual tracking initialization'
    )
    parser.add_argument(
        '--init-tip-pos',
        type=str,
        default=None,
        help='Initial tip position as "x,y" (e.g., "902,707") for manual tracking initialization'
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
        '--exclude-right-pixels',
        type=int,
        default=400,
        help='Exclude this many pixels from the right edge (for rulers/overlays) (default: 400)'
    )
    parser.add_argument(
        '--calculate-delta-gape',
        action='store_true',
        help='Calculate delta gape (change from initial position) in addition to absolute gape'
    )
    parser.add_argument(
        '--initial-gape',
        type=float,
        default=None,
        help='Known initial gape in mm (for delta gape calculation). Overrides auto-detection from first frame.'
    )
    parser.add_argument(
        '--use-datasheet-initial-gape',
        action='store_true',
        help='Automatically load initial gape from master datasheet based on video filename'
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video_path)
    csv_path = Path(args.csv_path)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse initial positions if provided
    init_shaft_pos = None
    init_tip_pos = None
    if args.init_shaft_pos:
        try:
            x, y = map(int, args.init_shaft_pos.split(','))
            init_shaft_pos = (x, y)
        except:
            print(f"Error: Invalid shaft position format: {args.init_shaft_pos}")
            print("Expected format: x,y (e.g., 724,460)")
            sys.exit(1)

    if args.init_tip_pos:
        try:
            x, y = map(int, args.init_tip_pos.split(','))
            init_tip_pos = (x, y)
        except:
            print(f"Error: Invalid tip position format: {args.init_tip_pos}")
            print("Expected format: x,y (e.g., 902,707)")
            sys.exit(1)

    # Parse HSV values
    try:
        hsv_lower = tuple(map(int, args.hsv_lower.split(',')))
        if len(hsv_lower) != 3:
            raise ValueError("HSV lower must have 3 values")
    except:
        print(f"Error: Invalid HSV lower format: {args.hsv_lower}")
        print("Expected format: H,S,V (e.g., 35,100,50)")
        sys.exit(1)

    try:
        hsv_upper = tuple(map(int, args.hsv_upper.split(',')))
        if len(hsv_upper) != 3:
            raise ValueError("HSV upper must have 3 values")
    except:
        print(f"Error: Invalid HSV upper format: {args.hsv_upper}")
        print("Expected format: H,S,V (e.g., 55,255,255)")
        sys.exit(1)

    # Determine initial gape for delta calculation
    initial_gape_mm = None
    if args.calculate_delta_gape:
        if args.initial_gape is not None:
            # User provided explicit initial gape
            initial_gape_mm = args.initial_gape
        elif args.use_datasheet_initial_gape:
            # Load from master datasheet
            sample_name = video_path.stem  # e.g., "A1" from "A1.mp4"
            initial_gape_mm = load_initial_gape_from_datasheet(sample_name)
            if initial_gape_mm is None:
                print(f"Warning: Could not load initial gape for {sample_name} from datasheet")
                print("         Delta gape will use first tracked frame as initial")

    print("="*70)
    print("PyGapeVision - Synchronized Analysis")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"CSV:   {csv_path}")
    print(f"Output: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"HSV range: {hsv_lower} to {hsv_upper}")
    if args.calculate_delta_gape:
        if initial_gape_mm is not None:
            print(f"Delta gape: Enabled (using measured initial gape: {initial_gape_mm} mm)")
        else:
            print(f"Delta gape: Enabled (using first tracked frame as initial)")
    if args.time_offset is not None:
        print(f"Time offset (manual): {args.time_offset} seconds")
    else:
        print("Time offset: Auto-sync enabled")
    if init_shaft_pos and init_tip_pos:
        print(f"Manual tracking initialization:")
        print(f"  Shaft: {init_shaft_pos}")
        print(f"  Tip:   {init_tip_pos}")
    print("="*70)

    # Step 1: Load mechanical data
    print("\n[1/4] Loading mechanical test data...")
    shimadzu_df = load_shimadzu_csv(str(csv_path))
    test_start_time = find_test_start(shimadzu_df)
    print(f"Mechanical test starts at t={test_start_time:.3f} seconds")
    print(f"Loaded {len(shimadzu_df)} data points from CSV")

    # Step 2: Initial video processing for time sync
    print("\n[2/4] Processing video for synchronization...")
    tracker = GreenPointTracker(hsv_lower=hsv_lower, hsv_upper=hsv_upper,
                                 exclude_right_pixels=args.exclude_right_pixels)
    analyzer = VideoAnalyzer(str(video_path), tracker)

    # First pass - just track points without saving video
    print("  First pass: Tracking points...")
    results = analyzer.process_video(
        output_video_path=None,
        frame_skip=args.frame_skip,
        fps_override=args.fps,
        init_shaft_pos=init_shaft_pos,
        init_tip_pos=init_tip_pos
    )

    # Check if tracking succeeded - retry with simple method if it failed
    frames_tracked = results.get('frames_tracked', 0)
    tracking_method = results.get('tracking_method', 'unknown')

    if frames_tracked < 10:
        print(f"  ⚠️  Warning: Only {frames_tracked} frames tracked with {tracking_method} method")
        print("  Retrying with simple tracking method (two largest contours)...")

        # Retry with simple tracking
        tracker_simple = GreenPointTracker(
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            use_simple_tracking=True,
            exclude_right_pixels=args.exclude_right_pixels
        )
        analyzer_simple = VideoAnalyzer(str(video_path), tracker_simple)
        results_retry = analyzer_simple.process_video(
            output_video_path=None,
            frame_skip=args.frame_skip,
            fps_override=args.fps,
            init_shaft_pos=None,  # Don't use manual init with simple tracking
            init_tip_pos=None
        )

        frames_tracked_retry = results_retry.get('frames_tracked', 0)

        if frames_tracked_retry > frames_tracked:
            print(f"  ✓ Simple tracking method succeeded: {frames_tracked_retry} frames tracked")
            results = results_retry
            tracker = tracker_simple
            frames_tracked = frames_tracked_retry
            tracking_method = results.get('tracking_method', 'simple')
        else:
            print(f"  ✗ Simple tracking also failed: {frames_tracked_retry} frames tracked")
            print(f"  Proceeding with original {tracking_method} results ({frames_tracked} frames)")
    else:
        print(f"  ✓ Tracking succeeded: {frames_tracked} frames tracked using {tracking_method} method")

    # Step 3: Synchronize data
    print("\n[3/4] Synchronizing video and mechanical data...")

    if args.time_offset is not None:
        # Manual offset
        time_offset = args.time_offset
        correlation = None
        print(f"Using manual time offset: {time_offset:.3f} seconds")
    else:
        # Auto-sync using peak alignment method
        print("Searching for optimal time alignment...")
        time_offset, correlation = auto_sync_video_to_mechanical(
            results,
            shimadzu_df,
            method='peak_alignment'
        )
        if correlation is not None:
            print(f"  Correlation: {correlation:.3f}")

    # Create synchronized dataset
    synced_df = sync_data(
        results,
        shimadzu_df,
        time_offset,
        calculate_delta=args.calculate_delta_gape,
        initial_gape_mm=initial_gape_mm,
        pixels_per_mm=args.pixels_per_mm
    )

    # Get video name for output files
    video_name = video_path.stem
    output_video_path = None

    # Step 3b: Create annotated video with overlays (if requested)
    if not args.no_video:
        print("\n  Second pass: Creating annotated video with overlays...")
        output_video_path = output_dir / f"{video_name}_tracked.mp4"

        # Re-run with video output and overlays using the successful tracker
        tracker2 = GreenPointTracker(
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            use_simple_tracking=tracker.use_simple_tracking,
            exclude_right_pixels=args.exclude_right_pixels
        )
        analyzer2 = VideoAnalyzer(str(video_path), tracker2)

        # Only use manual init if not using simple tracking
        shaft_init = init_shaft_pos if not tracker.use_simple_tracking else None
        tip_init = init_tip_pos if not tracker.use_simple_tracking else None

        analyzer2.process_video(
            output_video_path=str(output_video_path),
            frame_skip=args.frame_skip,
            fps_override=args.fps,
            rotate_90_cw=args.rotate,
            shimadzu_df=shimadzu_df,
            time_offset=time_offset,
            pixels_per_mm=args.pixels_per_mm,
            init_shaft_pos=shaft_init,
            init_tip_pos=tip_init
        )
        print(f"  Annotated video saved to: {output_video_path}")

    # Step 4: Generate outputs
    print("\n[4/4] Generating outputs...")

    # Save synchronized data
    output_csv_path = output_dir / f"{video_name}_synchronized.csv"
    synced_df.to_csv(output_csv_path, index=False)
    print(f"Synchronized CSV saved to: {output_csv_path}")

    # Create synchronized plot
    output_plot_path = output_dir / f"{video_name}_synchronized_analysis.png"
    plot_synchronized_data(
        synced_df,
        output_path=str(output_plot_path),
        pixels_per_mm=args.pixels_per_mm
    )

    # Create standard tracking plot
    output_track_plot = output_dir / f"{video_name}_tracking_analysis.png"
    plot_tracking_results(
        results,
        output_path=str(output_track_plot),
        pixels_per_mm=args.pixels_per_mm
    )

    # Create force vs gape plot
    output_force_gape_plot = output_dir / f"{video_name}_force_vs_gape.png"
    plot_force_vs_gape(
        synced_df,
        output_path=str(output_force_gape_plot),
        pixels_per_mm=args.pixels_per_mm
    )

    # Create stroke vs force plot
    output_stroke_force_plot = output_dir / f"{video_name}_stroke_vs_force.png"
    plot_stroke_vs_force(
        synced_df,
        output_path=str(output_stroke_force_plot)
    )

    # Save combined results (tracking data + sync metadata) to single JSON
    combined_results = {
        'metadata': {
            'video_path': str(video_path),
            'csv_path': str(csv_path),
            'video_fps': args.fps,
            'frame_skip': args.frame_skip,
            'pixels_per_mm': args.pixels_per_mm,
            'time_offset_seconds': time_offset,
            'correlation': correlation if correlation is not None else 'manual/peak_alignment',
            'test_start_time': test_start_time,
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper,
            'calculate_delta_gape': args.calculate_delta_gape,
            'initial_gape_px': float(synced_df['Initial_Gape_px'].iloc[0]) if args.calculate_delta_gape and synced_df['Initial_Gape_px'].notna().any() else None,
            'initial_gape_mm': initial_gape_mm,
            'initial_gape_source': 'manual' if args.initial_gape is not None else ('datasheet' if args.use_datasheet_initial_gape else 'first_frame'),
            'tracking_method': tracking_method,
            'frames_tracked': frames_tracked,
        },
        'tracking_data': results
    }

    output_json_path = output_dir / f"{video_name}_results.json"
    with open(output_json_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"Results JSON saved to: {output_json_path}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Synchronized data: {output_csv_path}")
    print(f"Synchronized plot: {output_plot_path}")
    print(f"Tracking plot: {output_track_plot}")
    print(f"Gape vs Force plot: {output_force_gape_plot}")
    print(f"Stroke vs Force plot: {output_stroke_force_plot}")
    print(f"Results JSON: {output_json_path}")
    if output_video_path:
        print(f"Annotated video: {output_video_path}")
    print("="*70)


if __name__ == "__main__":
    main()
