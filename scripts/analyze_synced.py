#!/usr/bin/env python
"""
Synchronized analysis combining video tracking with mechanical test data.

Usage (from project root):
    python scripts/analyze_synced.py <video_path> <csv_path> [options]
"""

import argparse
import sys
import json
import csv
import yaml
from pathlib import Path
from typing import Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import GreenPointTracker, VideoAnalyzer
from src.visualization import plot_tracking_results, plot_force_vs_gape, plot_stroke_vs_force
from src.sync import (load_shimadzu_csv, find_test_start, auto_sync_video_to_mechanical,
                     sync_data, plot_synchronized_data)
from src.datasheet import load_initial_gape_from_datasheet


def load_sync_overrides(config_path: Path) -> tuple[dict, dict]:
    defaults = {}
    overrides = {}
    if not config_path.exists():
        return defaults, overrides

    with config_path.open(newline="") as f:
        rows = (row for row in f if row.strip() and not row.lstrip().startswith("#"))
        reader = csv.DictReader(rows)
        for row in reader:
            sample = (row.get("sample") or row.get("Sample") or "").strip()
            if not sample:
                continue
            if sample.lower() in ("default", "*"):
                defaults = row
                continue
            overrides[sample] = row
    return defaults, overrides


def apply_sync_overrides(args, override_row: dict) -> None:
    field_types = {
        "sync_method": str,
        "drop_smooth_window": int,
        "sync_search_min": float,
        "sync_search_max": float,
        "sync_search_steps": int,
        "signature_force_weight": float,
        "signature_stroke_weight": float,
        "signature_smooth_window": int,
        "force_threshold": float,
        "stroke_threshold": float,
        "video_change_threshold_px": float,
        "baseline_frames": int,
        "min_consecutive": int,
    }

    for field, caster in field_types.items():
        raw_value = override_row.get(field)
        if raw_value is None:
            continue
        raw_value = raw_value.strip()
        if raw_value == "":
            continue
        try:
            setattr(args, field, caster(raw_value))
        except ValueError:
            print(f"Warning: Invalid override for {field}: {raw_value}")


def load_processing_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def _format_hsv(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return ",".join(str(int(v)) for v in value)
    return str(value)


def _format_point(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])},{int(value[1])}"
    return str(value)

def _format_bbox(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return ",".join(str(int(v)) for v in value)
    return str(value)

def _parse_int_list(value, count: int) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == count:
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) == count:
            return [int(p.strip()) for p in parts]
    return None


def apply_config_overrides(args, override_block: dict) -> None:
    if not override_block:
        return

    sync = override_block.get("sync", {})
    analysis = override_block.get("analysis", {})
    tracking = override_block.get("tracking", {})

    sync_map = {
        "method": "sync_method",
        "drop_smooth_window": "drop_smooth_window",
        "search_min": "sync_search_min",
        "search_max": "sync_search_max",
        "search_steps": "sync_search_steps",
        "signature_force_weight": "signature_force_weight",
        "signature_stroke_weight": "signature_stroke_weight",
        "signature_smooth_window": "signature_smooth_window",
        "force_threshold": "force_threshold",
        "stroke_threshold": "stroke_threshold",
        "video_change_threshold_px": "video_change_threshold_px",
        "baseline_frames": "baseline_frames",
        "min_consecutive": "min_consecutive",
    }

    tracking_map = {
        "exclude_right_pixels": "exclude_right_pixels",
        "use_simple_tracking": "use_simple_tracking",
        "min_area": "min_area",
        "max_area": "max_area",
        "max_x_ratio": "max_x_ratio",
        "search_radius": "search_radius",
        "validate_tip_edges": "validate_tip_edges",
        "min_edge_separation": "min_edge_separation",
        "min_contour_area": "min_contour_area",
        "morph_kernel_size": "morph_kernel_size",
        "tip_min_area": "tip_min_area",
        "tip_max_area": "tip_max_area",
        "tip_min_contour_area": "tip_min_contour_area",
        "tip_morph_kernel_size": "tip_morph_kernel_size",
        "tip_exclude_right_pixels": "tip_exclude_right_pixels",
        "tip_max_x_ratio": "tip_max_x_ratio",
        "tip_roi_mode": "tip_roi_mode",
        "tip_roi_radius": "tip_roi_radius",
    }

    analysis_map = {
        "frame_skip": "frame_skip",
        "fps": "fps",
        "pixels_per_mm": "pixels_per_mm",
        "calculate_delta_gape": "calculate_delta_gape",
        "use_datasheet_initial_gape": "use_datasheet_initial_gape",
        "show_true_39mm": "show_true_39mm",
        "initial_gape": "initial_gape",
        "time_offset": "time_offset",
    }

    for key, attr in sync_map.items():
        if key in sync:
            setattr(args, attr, sync[key])

    for key, attr in analysis_map.items():
        if key in analysis:
            setattr(args, attr, analysis[key])

    if "hsv_lower" in tracking:
        args.hsv_lower = _format_hsv(tracking["hsv_lower"])
    if "hsv_upper" in tracking:
        args.hsv_upper = _format_hsv(tracking["hsv_upper"])
    if "tip_hsv_lower" in tracking:
        args.tip_hsv_lower = _format_hsv(tracking["tip_hsv_lower"])
    if "tip_hsv_upper" in tracking:
        args.tip_hsv_upper = _format_hsv(tracking["tip_hsv_upper"])
    if "init_shaft_pos" in tracking:
        args.init_shaft_pos = _format_point(tracking["init_shaft_pos"])
    if "init_tip_pos" in tracking:
        args.init_tip_pos = _format_point(tracking["init_tip_pos"])
    if "tip_roi" in tracking:
        args.tip_roi = _format_bbox(tracking["tip_roi"])
    if "tip_roi_mode" in tracking:
        args.tip_roi_mode = tracking["tip_roi_mode"]
    if "tip_roi_radius" in tracking:
        args.tip_roi_radius = tracking["tip_roi_radius"]
    if "reacquire_points" in tracking:
        args.reacquire_points = tracking["reacquire_points"]

    for key, attr in tracking_map.items():
        if key in tracking:
            setattr(args, attr, tracking[key])


def save_tracking_to_config(config_path: Path, sample_name: str,
                            init_shaft_pos: Tuple[int, int],
                            init_tip_pos: Tuple[int, int],
                            reacquire_points: list[dict],
                            args: argparse.Namespace) -> None:
    config_data = load_processing_config(config_path)
    if not config_data:
        config_data = {}
    samples = config_data.setdefault("samples", {})
    sample_block = samples.setdefault(sample_name, {})
    analysis = sample_block.setdefault("analysis", {})
    sync = sample_block.setdefault("sync", {})
    tracking = sample_block.setdefault("tracking", {})

    analysis.update({
        "frame_skip": args.frame_skip,
        "fps": args.fps,
        "pixels_per_mm": args.pixels_per_mm,
        "calculate_delta_gape": args.calculate_delta_gape,
        "use_datasheet_initial_gape": args.use_datasheet_initial_gape,
        "show_true_39mm": args.show_true_39mm,
    })
    if args.initial_gape is not None:
        analysis["initial_gape"] = args.initial_gape
    if args.time_offset is not None:
        analysis["time_offset"] = args.time_offset

    sync.update({
        "method": args.sync_method,
        "drop_smooth_window": args.drop_smooth_window,
        "search_min": args.sync_search_min,
        "search_max": args.sync_search_max,
        "search_steps": args.sync_search_steps,
        "signature_force_weight": args.signature_force_weight,
        "signature_stroke_weight": args.signature_stroke_weight,
        "signature_smooth_window": args.signature_smooth_window,
        "force_threshold": args.force_threshold,
        "stroke_threshold": args.stroke_threshold,
        "video_change_threshold_px": args.video_change_threshold_px,
        "baseline_frames": args.baseline_frames,
        "min_consecutive": args.min_consecutive,
    })

    tracking["init_shaft_pos"] = [int(init_shaft_pos[0]), int(init_shaft_pos[1])]
    tracking["init_tip_pos"] = [int(init_tip_pos[0]), int(init_tip_pos[1])]
    tracking["exclude_right_pixels"] = args.exclude_right_pixels
    tracking["use_simple_tracking"] = args.use_simple_tracking
    tracking["min_area"] = args.min_area
    tracking["max_area"] = args.max_area
    tracking["max_x_ratio"] = args.max_x_ratio
    tracking["search_radius"] = args.search_radius
    tracking["validate_tip_edges"] = args.validate_tip_edges
    tracking["min_edge_separation"] = args.min_edge_separation
    tracking["min_contour_area"] = args.min_contour_area
    tracking["morph_kernel_size"] = args.morph_kernel_size
    tracking["tip_min_area"] = args.tip_min_area
    tracking["tip_max_area"] = args.tip_max_area
    tracking["tip_min_contour_area"] = args.tip_min_contour_area
    tracking["tip_morph_kernel_size"] = args.tip_morph_kernel_size
    tracking["tip_exclude_right_pixels"] = args.tip_exclude_right_pixels
    tracking["tip_max_x_ratio"] = args.tip_max_x_ratio
    tracking["tip_roi_mode"] = args.tip_roi_mode
    tracking["tip_roi_radius"] = args.tip_roi_radius

    hsv_lower = _parse_int_list(args.hsv_lower, 3)
    hsv_upper = _parse_int_list(args.hsv_upper, 3)
    if hsv_lower:
        tracking["hsv_lower"] = hsv_lower
    if hsv_upper:
        tracking["hsv_upper"] = hsv_upper
    if args.tip_hsv_lower:
        tip_lower = _parse_int_list(args.tip_hsv_lower, 3)
        if tip_lower:
            tracking["tip_hsv_lower"] = tip_lower
    if args.tip_hsv_upper:
        tip_upper = _parse_int_list(args.tip_hsv_upper, 3)
        if tip_upper:
            tracking["tip_hsv_upper"] = tip_upper
    if args.tip_roi:
        tip_roi = _parse_int_list(args.tip_roi, 4)
        if tip_roi:
            tracking["tip_roi"] = tip_roi

    if reacquire_points:
        tracking["reacquire_points"] = [
            {
                "frame_number": int(point["frame_number"]),
                "shaft_pos": [int(point["shaft_pos"][0]), int(point["shaft_pos"][1])],
                "tip_pos": [int(point["tip_pos"][0]), int(point["tip_pos"][1])],
            }
            for point in reacquire_points
            if point.get("frame_number") is not None
            and point.get("shaft_pos")
            and point.get("tip_pos")
        ]
    with config_path.open("w") as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
    print(f"Saved tracking init points to config: {config_path} ({sample_name})")


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
        '--sync-method',
        type=str,
        default='start_alignment',
        choices=['start_alignment', 'peak_alignment', 'drop_alignment', 'multi_signature', 'correlation'],
        help='Auto-sync method: start_alignment, peak_alignment, drop_alignment, multi_signature, or correlation (default: start_alignment)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='YAML config with defaults and per-sample overrides'
    )
    parser.add_argument(
        '--sync-config',
        type=str,
        default=None,
        help='Optional CSV mapping sample to sync overrides (e.g., configs/sync_overrides.csv)'
    )
    parser.add_argument(
        '--sync-search-min',
        type=float,
        default=-30.0,
        help='Minimum time offset (s) for multi_signature/correlation (default: -30)'
    )
    parser.add_argument(
        '--sync-search-max',
        type=float,
        default=30.0,
        help='Maximum time offset (s) for multi_signature/correlation (default: 30)'
    )
    parser.add_argument(
        '--sync-search-steps',
        type=int,
        default=200,
        help='Number of offsets to evaluate for multi_signature/correlation (default: 200)'
    )
    parser.add_argument(
        '--force-threshold',
        type=float,
        default=0.1,
        help='Force threshold (N) for start_alignment (default: 0.1)'
    )
    parser.add_argument(
        '--stroke-threshold',
        type=float,
        default=0.01,
        help='Stroke threshold (mm) for start_alignment (default: 0.01)'
    )
    parser.add_argument(
        '--video-change-threshold-px',
        type=float,
        default=3.0,
        help='Pixel change threshold for start_alignment (default: 3.0)'
    )
    parser.add_argument(
        '--baseline-frames',
        type=int,
        default=5,
        help='Baseline frame count for start_alignment (default: 5)'
    )
    parser.add_argument(
        '--min-consecutive',
        type=int,
        default=3,
        help='Minimum consecutive frames above threshold for start_alignment (default: 3)'
    )
    parser.add_argument(
        '--drop-smooth-window',
        type=int,
        default=5,
        help='Smoothing window for drop_alignment (default: 5)'
    )
    parser.add_argument(
        '--signature-force-weight',
        type=float,
        default=0.7,
        help='Force rate weight for multi_signature (default: 0.7)'
    )
    parser.add_argument(
        '--signature-stroke-weight',
        type=float,
        default=0.3,
        help='Stroke rate weight for multi_signature (default: 0.3)'
    )
    parser.add_argument(
        '--signature-smooth-window',
        type=int,
        default=5,
        help='Smoothing window for multi_signature (default: 5)'
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
        '--interactive-init',
        action='store_true',
        help='Prompt for manual points on the first processed frame'
    )
    parser.add_argument(
        '--interactive-reacquire',
        action='store_true',
        help='Prompt for manual points when tracking is lost'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable all interactive prompts (use config points only)'
    )
    parser.add_argument(
        '--save-tracking-to-config',
        action='store_true',
        help='Save initial tracking points to the YAML config (requires --config)'
    )
    parser.add_argument(
        '--reacquire-miss-frames',
        type=int,
        default=15,
        help='Consecutive missed frames before prompting (default: 15)'
    )
    parser.add_argument(
        '--reacquire-max-retries',
        type=int,
        default=5,
        help='Maximum manual reacquire prompts per video (default: 5)'
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
        '--tip-hsv-lower',
        type=str,
        default=None,
        help='HSV lower bound for tip detection (H,S,V). Defaults to --hsv-lower if not set.'
    )
    parser.add_argument(
        '--tip-hsv-upper',
        type=str,
        default=None,
        help='HSV upper bound for tip detection (H,S,V). Defaults to --hsv-upper if not set.'
    )
    parser.add_argument(
        '--tip-roi',
        type=str,
        default=None,
        help='Tip ROI as "x1,y1,x2,y2" (limits tip detection region)'
    )
    parser.add_argument(
        '--tip-roi-mode',
        type=str,
        default="static",
        choices=["static", "initial_only", "follow_prev"],
        help='Tip ROI behavior: static, initial_only, or follow_prev (default: static)'
    )
    parser.add_argument(
        '--tip-roi-radius',
        type=int,
        default=None,
        help='Tip ROI radius (pixels) when using follow_prev (default: --search-radius)'
    )
    parser.add_argument(
        '--exclude-right-pixels',
        type=int,
        default=400,
        help='Exclude this many pixels from the right edge (for rulers/overlays) (default: 400)'
    )
    parser.add_argument(
        '--use-simple-tracking',
        action='store_true',
        help='Use simple tracking (two largest contours)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=200,
        help='Minimum contour area to consider (default: 200)'
    )
    parser.add_argument(
        '--max-area',
        type=int,
        default=15000,
        help='Maximum contour area to consider (default: 15000)'
    )
    parser.add_argument(
        '--tip-min-area',
        type=int,
        default=None,
        help='Minimum contour area for tip candidates (default: --min-area)'
    )
    parser.add_argument(
        '--tip-max-area',
        type=int,
        default=None,
        help='Maximum contour area for tip candidates (default: --max-area)'
    )
    parser.add_argument(
        '--max-x-ratio',
        type=float,
        default=0.65,
        help='Max X ratio for initial detection region (default: 0.65)'
    )
    parser.add_argument(
        '--search-radius',
        type=int,
        default=150,
        help='Search radius in pixels for temporal tracking (default: 150)'
    )
    parser.add_argument(
        '--min-contour-area',
        type=int,
        default=100,
        help='Minimum contour area for green regions (default: 100)'
    )
    parser.add_argument(
        '--tip-min-contour-area',
        type=int,
        default=None,
        help='Minimum contour area for tip regions (default: --min-contour-area)'
    )
    parser.add_argument(
        '--morph-kernel-size',
        type=int,
        default=5,
        help='Morphological kernel size for mask cleanup (default: 5)'
    )
    parser.add_argument(
        '--tip-morph-kernel-size',
        type=int,
        default=None,
        help='Morphological kernel size for tip mask cleanup (default: --morph-kernel-size)'
    )
    parser.add_argument(
        '--no-validate-tip-edges',
        dest='validate_tip_edges',
        action='store_false',
        help='Disable edge-based tip validation (useful for blurred tips)'
    )
    parser.set_defaults(validate_tip_edges=True)
    parser.add_argument(
        '--min-edge-separation',
        type=int,
        default=15,
        help='Minimum edge separation for tip validation (default: 15)'
    )
    parser.add_argument(
        '--tip-exclude-right-pixels',
        type=int,
        default=None,
        help='Exclude this many pixels from right edge for tip detection (default: --exclude-right-pixels)'
    )
    parser.add_argument(
        '--tip-max-x-ratio',
        type=float,
        default=None,
        help='Max X ratio for tip candidates (default: --max-x-ratio)'
    )
    parser.add_argument(
        '--calculate-delta-gape',
        action='store_true',
        help='Calculate delta gape (change from initial position) in addition to absolute gape'
    )
    parser.add_argument(
        '--show-true-39mm',
        action='store_true',
        help='Show true 39mm threshold using measured initial gape (requires delta + pixels-per-mm)'
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
    args.reacquire_points = None

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

    # Apply per-sample config overrides if provided
    applied_override = False
    applied_default = False
    if args.config:
        config_path = Path(args.config)
        config_data = load_processing_config(config_path)
        if config_data:
            defaults = config_data.get("defaults", {})
            if defaults:
                apply_config_overrides(args, defaults)
                applied_default = True
                print(f"Applied config defaults from {config_path}")
            samples = config_data.get("samples", {})
            sample_override = samples.get(video_path.stem)
            if sample_override:
                apply_config_overrides(args, sample_override)
                applied_override = True
                print(f"Applied config overrides from {config_path} for sample {video_path.stem}")
        else:
            print(f"Warning: Config file is empty or invalid: {config_path}")
    elif args.sync_config:
        config_path = Path(args.sync_config)
        defaults, overrides = load_sync_overrides(config_path)
        if defaults:
            apply_sync_overrides(args, defaults)
            applied_default = True
            print(f"Applied sync defaults from {config_path}")
        override_row = overrides.get(video_path.stem)
        if override_row:
            apply_sync_overrides(args, override_row)
            applied_override = True
            print(f"Applied sync overrides from {config_path} for sample {video_path.stem}")

    if args.no_interactive:
        args.interactive_init = False
        args.interactive_reacquire = False

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

    tip_hsv_lower = None
    if args.tip_hsv_lower:
        try:
            tip_hsv_lower = tuple(map(int, args.tip_hsv_lower.split(',')))
            if len(tip_hsv_lower) != 3:
                raise ValueError("Tip HSV lower must have 3 values")
        except:
            print(f"Error: Invalid tip HSV lower format: {args.tip_hsv_lower}")
            print("Expected format: H,S,V (e.g., 44,24,158)")
            sys.exit(1)

    tip_hsv_upper = None
    if args.tip_hsv_upper:
        try:
            tip_hsv_upper = tuple(map(int, args.tip_hsv_upper.split(',')))
            if len(tip_hsv_upper) != 3:
                raise ValueError("Tip HSV upper must have 3 values")
        except:
            print(f"Error: Invalid tip HSV upper format: {args.tip_hsv_upper}")
            print("Expected format: H,S,V (e.g., 70,255,255)")
            sys.exit(1)

    tip_roi = None
    if args.tip_roi:
        try:
            tip_roi = tuple(map(int, args.tip_roi.split(',')))
            if len(tip_roi) != 4:
                raise ValueError("Tip ROI must have 4 values")
            if tip_roi[2] <= tip_roi[0] or tip_roi[3] <= tip_roi[1]:
                raise ValueError("Tip ROI must have x2>x1 and y2>y1")
        except:
            print(f"Error: Invalid tip ROI format: {args.tip_roi}")
            print("Expected format: x1,y1,x2,y2 (e.g., 880,620,980,740)")
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
    if tip_hsv_lower or tip_hsv_upper:
        tip_lower = tip_hsv_lower if tip_hsv_lower is not None else hsv_lower
        tip_upper = tip_hsv_upper if tip_hsv_upper is not None else hsv_upper
        print(f"Tip HSV range: {tip_lower} to {tip_upper}")
    if tip_roi:
        print(f"Tip ROI: {tip_roi}")
        print(f"Tip ROI mode: {args.tip_roi_mode}")
        if args.tip_roi_mode == "follow_prev":
            radius = args.tip_roi_radius if args.tip_roi_radius is not None else args.search_radius
            print(f"Tip ROI radius: {radius}")
    if args.calculate_delta_gape:
        if initial_gape_mm is not None:
            print(f"Delta gape: Enabled (using measured initial gape: {initial_gape_mm} mm)")
        else:
            print(f"Delta gape: Enabled (using first tracked frame as initial)")
        if args.show_true_39mm and (initial_gape_mm is None or args.pixels_per_mm is None):
            print("Warning: --show-true-39mm requires --use-datasheet-initial-gape and --pixels-per-mm")
    elif args.show_true_39mm:
        print("Warning: --show-true-39mm requires --calculate-delta-gape")
    if args.time_offset is not None:
        print(f"Time offset (manual): {args.time_offset} seconds")
    else:
        print(f"Time offset: Auto-sync enabled ({args.sync_method})")
        if args.sync_method == 'start_alignment':
            print(
                f"  Start thresholds: force>{args.force_threshold} N, "
                f"stroke>{args.stroke_threshold} mm, "
                f"video_change>{args.video_change_threshold_px} px"
            )
        if args.sync_method == 'drop_alignment':
            print(f"  Drop smoothing window: {args.drop_smooth_window}")
        if args.sync_method == 'multi_signature':
            print(
                f"  Signature weights: force={args.signature_force_weight}, "
                f"stroke={args.signature_stroke_weight}, "
                f"smooth_window={args.signature_smooth_window}"
            )
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
                                 use_simple_tracking=args.use_simple_tracking,
                                 exclude_right_pixels=args.exclude_right_pixels,
                                 min_area=args.min_area,
                                 max_area=args.max_area,
                                 max_x_ratio=args.max_x_ratio,
                                 search_radius=args.search_radius,
                                 validate_tip_edges=args.validate_tip_edges,
                                 min_edge_separation=args.min_edge_separation,
                                 min_contour_area=args.min_contour_area,
                                 morph_kernel_size=args.morph_kernel_size,
                                 tip_hsv_lower=tip_hsv_lower,
                                 tip_hsv_upper=tip_hsv_upper,
                                 tip_roi=tip_roi,
                                 tip_roi_mode=args.tip_roi_mode,
                                 tip_roi_radius=args.tip_roi_radius,
                                 tip_min_area=args.tip_min_area,
                                 tip_max_area=args.tip_max_area,
                                 tip_min_contour_area=args.tip_min_contour_area,
                                 tip_morph_kernel_size=args.tip_morph_kernel_size,
                                 tip_exclude_right_pixels=args.tip_exclude_right_pixels,
                                 tip_max_x_ratio=args.tip_max_x_ratio)
    analyzer = VideoAnalyzer(str(video_path), tracker)

    # First pass - just track points without saving video
    print("  First pass: Tracking points...")
    results = analyzer.process_video(
        output_video_path=None,
        frame_skip=args.frame_skip,
        fps_override=args.fps,
        init_shaft_pos=init_shaft_pos,
        init_tip_pos=init_tip_pos,
        interactive_init=args.interactive_init,
        interactive_reacquire=args.interactive_reacquire,
        reacquire_miss_frames=args.reacquire_miss_frames,
        reacquire_max_retries=args.reacquire_max_retries,
        reacquire_points=args.reacquire_points
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
            exclude_right_pixels=args.exclude_right_pixels,
            min_area=args.min_area,
            max_area=args.max_area,
            max_x_ratio=args.max_x_ratio,
            search_radius=args.search_radius,
            validate_tip_edges=args.validate_tip_edges,
            min_edge_separation=args.min_edge_separation,
            min_contour_area=args.min_contour_area,
            morph_kernel_size=args.morph_kernel_size,
            tip_hsv_lower=tip_hsv_lower,
            tip_hsv_upper=tip_hsv_upper,
            tip_roi=tip_roi,
            tip_roi_mode=args.tip_roi_mode,
            tip_roi_radius=args.tip_roi_radius,
            tip_min_area=args.tip_min_area,
            tip_max_area=args.tip_max_area,
            tip_min_contour_area=args.tip_min_contour_area,
            tip_morph_kernel_size=args.tip_morph_kernel_size,
            tip_exclude_right_pixels=args.tip_exclude_right_pixels,
            tip_max_x_ratio=args.tip_max_x_ratio
        )
        analyzer_simple = VideoAnalyzer(str(video_path), tracker_simple)
        results_retry = analyzer_simple.process_video(
            output_video_path=None,
            frame_skip=args.frame_skip,
            fps_override=args.fps,
            init_shaft_pos=None,  # Don't use manual init with simple tracking
            init_tip_pos=None,
            interactive_init=args.interactive_init,
            interactive_reacquire=args.interactive_reacquire,
            reacquire_miss_frames=args.reacquire_miss_frames,
            reacquire_max_retries=args.reacquire_max_retries,
            reacquire_points=args.reacquire_points
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
        print(f"Searching for optimal time alignment ({args.sync_method})...")
        search_range = (args.sync_search_min, args.sync_search_max)
        time_offset, correlation = auto_sync_video_to_mechanical(
            results,
            shimadzu_df,
            method=args.sync_method,
            search_range=search_range,
            search_steps=args.sync_search_steps,
            force_threshold=args.force_threshold,
            stroke_threshold=args.stroke_threshold,
            video_change_threshold_px=args.video_change_threshold_px,
            baseline_frames=args.baseline_frames,
            min_consecutive=args.min_consecutive,
            drop_smooth_window=args.drop_smooth_window,
            signature_force_weight=args.signature_force_weight,
            signature_stroke_weight=args.signature_stroke_weight,
            signature_smooth_window=args.signature_smooth_window
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
            exclude_right_pixels=args.exclude_right_pixels,
            min_area=args.min_area,
            max_area=args.max_area,
            max_x_ratio=args.max_x_ratio,
            search_radius=args.search_radius,
            validate_tip_edges=args.validate_tip_edges,
            min_edge_separation=args.min_edge_separation,
            min_contour_area=args.min_contour_area,
            morph_kernel_size=args.morph_kernel_size,
            tip_hsv_lower=tip_hsv_lower,
            tip_hsv_upper=tip_hsv_upper,
            tip_roi=tip_roi,
            tip_roi_mode=args.tip_roi_mode,
            tip_roi_radius=args.tip_roi_radius,
            tip_min_area=args.tip_min_area,
            tip_max_area=args.tip_max_area,
            tip_min_contour_area=args.tip_min_contour_area,
            tip_morph_kernel_size=args.tip_morph_kernel_size,
            tip_exclude_right_pixels=args.tip_exclude_right_pixels,
            tip_max_x_ratio=args.tip_max_x_ratio
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
            init_tip_pos=tip_init,
            interactive_init=False,
            interactive_reacquire=False,
            reacquire_points=results.get('reacquire_points'),
            tracking_override=results,
            display_delta_mm=args.calculate_delta_gape,
            initial_gape_mm=initial_gape_mm if args.show_true_39mm else None,
            show_true_39mm=args.show_true_39mm
        )
        print(f"  Annotated video saved to: {output_video_path}")

    # Step 4: Generate outputs
    print("\n[4/4] Generating outputs...")

    if args.save_tracking_to_config:
        if not args.config:
            print("Warning: --save-tracking-to-config requires --config")
        else:
            saved_shaft = init_shaft_pos
            saved_tip = init_tip_pos
            if args.interactive_init and results.get("reacquire_points"):
                first_point = results["reacquire_points"][0]
                saved_shaft = tuple(first_point.get("shaft_pos", [])) or saved_shaft
                saved_tip = tuple(first_point.get("tip_pos", [])) or saved_tip
            if saved_shaft and saved_tip:
                save_tracking_to_config(
                    Path(args.config),
                    video_path.stem,
                    saved_shaft,
                    saved_tip,
                    results.get("reacquire_points", []),
                    args
                )
            else:
                print("Warning: No initial tracking points available to save")

    # Save synchronized data
    output_csv_path = output_dir / f"{video_name}_synchronized.csv"
    synced_df.to_csv(output_csv_path, index=False)
    print(f"Synchronized CSV saved to: {output_csv_path}")

    # Create synchronized plot
    output_plot_path = output_dir / f"{video_name}_synchronized_analysis.png"
    plot_synchronized_data(
        synced_df,
        output_path=str(output_plot_path),
        pixels_per_mm=args.pixels_per_mm,
        show_true_39mm=args.show_true_39mm
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
            'correlation': correlation,
            'sync_method': 'manual' if args.time_offset is not None else args.sync_method,
            'sync_force_threshold': args.force_threshold,
            'sync_stroke_threshold': args.stroke_threshold,
            'sync_video_change_threshold_px': args.video_change_threshold_px,
            'sync_baseline_frames': args.baseline_frames,
            'sync_min_consecutive': args.min_consecutive,
            'sync_drop_smooth_window': args.drop_smooth_window,
            'sync_search_min': args.sync_search_min,
            'sync_search_max': args.sync_search_max,
            'sync_search_steps': args.sync_search_steps,
            'sync_signature_force_weight': args.signature_force_weight,
            'sync_signature_stroke_weight': args.signature_stroke_weight,
            'sync_signature_smooth_window': args.signature_smooth_window,
            'show_true_39mm': args.show_true_39mm,
            'config_path': str(args.config) if args.config else None,
            'config_override_applied': applied_override if args.config else False,
            'config_default_applied': applied_default if args.config else False,
            'sync_config': str(args.sync_config) if args.sync_config else None,
            'sync_override_applied': applied_override if args.sync_config else False,
            'sync_default_applied': applied_default if args.sync_config else False,
            'test_start_time': test_start_time,
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper,
            'tip_hsv_lower': tip_hsv_lower,
            'tip_hsv_upper': tip_hsv_upper,
            'tip_roi': tip_roi,
            'tip_roi_mode': args.tip_roi_mode,
            'tip_roi_radius': args.tip_roi_radius,
            'tip_min_area': args.tip_min_area,
            'tip_max_area': args.tip_max_area,
            'tip_min_contour_area': args.tip_min_contour_area,
            'tip_morph_kernel_size': args.tip_morph_kernel_size,
            'tip_exclude_right_pixels': args.tip_exclude_right_pixels,
            'tip_max_x_ratio': args.tip_max_x_ratio,
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
