#!/usr/bin/env python
"""
Validate synchronized CSV outputs for alignment and tracking quality.

Checks:
- Required columns present
- Time monotonic increasing
- Sufficient tracked frames and acceptable NaN ratio
- Alignment of force peak vs gape peak
- Alignment of force drop vs gape drop
- Video start vs mechanical start timing
- Tracking completeness near end of test
- Basic tracking stability (std and step size)
"""

import argparse
import csv
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


def smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")


def compute_drop_time(time: np.ndarray, series: np.ndarray, window: int) -> Optional[float]:
    if len(series) < 2:
        return None
    smoothed = smooth_series(series, window)
    diff = np.diff(smoothed)
    if len(diff) == 0:
        return None
    drop_idx = np.argmin(diff)
    return float(time[1:][drop_idx])


def compute_mech_start_time(df: pd.DataFrame,
                            force_threshold: float,
                            stroke_threshold: float) -> Optional[float]:
    if "Force" not in df.columns or "Stroke" not in df.columns or "Time" not in df.columns:
        return None
    stroke_diff = df["Stroke"].diff()
    mask = (df["Force"] > force_threshold) & (stroke_diff > stroke_threshold / 10.0)
    if mask.any():
        return float(df.loc[mask.idxmax(), "Time"])
    return float(df["Time"].iloc[0])


def choose_gape_series(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    if "Gape_Distance_mm_corrected" in df.columns and df["Gape_Distance_mm_corrected"].notna().any():
        return df["Gape_Distance_mm_corrected"], "mm_corrected"
    if "Gape_Distance_mm" in df.columns and df["Gape_Distance_mm"].notna().any():
        return df["Gape_Distance_mm"], "mm"
    if "Gape_Distance_px" in df.columns and df["Gape_Distance_px"].notna().any():
        return df["Gape_Distance_px"], "px"
    return None, None


def validate_csv(path: Path, args: argparse.Namespace) -> Dict[str, object]:
    issue_codes: List[str] = []
    issue_details: List[str] = []
    status = "PASS"

    def add_issue(code: str, detail: str, severity: str) -> None:
        nonlocal status
        issue_codes.append(code)
        issue_details.append(detail)
        if severity == "FAIL":
            status = "FAIL"
        elif severity == "WARN" and status == "PASS":
            status = "WARN"

    df = pd.read_csv(path)

    required_cols = {"Time", "Force", "Stroke"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        return {
            "sample": path.stem.replace("_synchronized", ""),
            "path": str(path),
            "status": "FAIL",
            "issues": f"missing columns: {missing}",
            "issue_codes": "missing_columns",
        }

    time = df["Time"].values
    if not np.all(np.diff(time) > 0):
        add_issue(
            "time_not_monotonic",
            "time not strictly increasing",
            "FAIL"
        )

    gape_series, gape_units = choose_gape_series(df)
    if gape_series is None:
        add_issue(
            "missing_gape_series",
            "no usable gape series (mm_corrected/mm/px)",
            "FAIL"
        )
        return {
            "sample": path.stem.replace("_synchronized", ""),
            "path": str(path),
            "status": status,
            "issues": "; ".join(issue_details),
            "issue_codes": ";".join(issue_codes),
        }

    valid_mask = gape_series.notna()
    frames_tracked = int(valid_mask.sum())
    total_frames = int(len(gape_series))
    nan_ratio = 1.0 - (frames_tracked / max(total_frames, 1))

    if frames_tracked < args.min_frames:
        add_issue(
            "low_frames",
            f"tracked frames {frames_tracked} < min {args.min_frames}",
            "FAIL"
        )
    if nan_ratio > args.max_nan_ratio:
        add_issue(
            "high_nan_ratio",
            f"NaN ratio {nan_ratio:.2f} > max {args.max_nan_ratio:.2f}",
            "WARN"
        )

    gape_valid = gape_series[valid_mask].values
    time_valid = df.loc[valid_mask, "Time"].values

    if len(gape_valid) > 1:
        gape_std = float(np.nanstd(gape_valid))
        max_step = float(np.nanmax(np.abs(np.diff(gape_valid))))
    else:
        gape_std = 0.0
        max_step = 0.0

    if gape_std < args.min_gape_std:
        add_issue(
            "low_gape_std",
            f"gape std {gape_std:.2f} < min {args.min_gape_std:.2f}",
            "WARN"
        )
    if args.max_step > 0 and max_step > args.max_step:
        add_issue(
            "large_step",
            f"max step {max_step:.2f} > max {args.max_step:.2f}",
            "WARN"
        )

    t_force_peak = float(df.loc[df["Force"].idxmax(), "Time"])
    t_gape_peak = float(time_valid[np.argmax(gape_valid)]) if len(gape_valid) else np.nan
    peak_diff = abs(t_force_peak - t_gape_peak) if np.isfinite(t_gape_peak) else np.nan
    if np.isfinite(peak_diff) and peak_diff > args.max_peak_diff_s:
        add_issue(
            "peak_misaligned",
            f"peak diff {peak_diff:.2f}s > max {args.max_peak_diff_s:.2f}s",
            "WARN"
        )

    t_force_drop = compute_drop_time(time, df["Force"].values, args.drop_smooth_window)
    t_gape_drop = compute_drop_time(time_valid, gape_valid, args.drop_smooth_window)
    drop_diff = abs(t_force_drop - t_gape_drop) if t_force_drop and t_gape_drop else np.nan
    if np.isfinite(drop_diff) and drop_diff > args.max_drop_diff_s:
        add_issue(
            "drop_misaligned",
            f"drop diff {drop_diff:.2f}s > max {args.max_drop_diff_s:.2f}s",
            "WARN"
        )

    mech_start = compute_mech_start_time(df, args.force_threshold, args.stroke_threshold)
    t_first_gape = float(time_valid[0]) if len(time_valid) else np.nan
    start_diff = abs(t_first_gape - mech_start) if mech_start is not None else np.nan
    if np.isfinite(start_diff) and start_diff > args.max_start_diff_s:
        add_issue(
            "start_misaligned",
            f"start diff {start_diff:.2f}s > max {args.max_start_diff_s:.2f}s",
            "WARN"
        )

    last_gape_time = float(time_valid[-1]) if len(time_valid) else np.nan
    end_gap = float(time[-1] - last_gape_time) if np.isfinite(last_gape_time) else np.nan
    if np.isfinite(end_gap) and end_gap > args.max_end_gap_s:
        add_issue(
            "tracking_ended_early",
            f"end gap {end_gap:.2f}s > max {args.max_end_gap_s:.2f}s",
            "WARN"
        )

    return {
        "sample": path.stem.replace("_synchronized", ""),
        "path": str(path),
        "status": status,
        "issues": "; ".join(issue_details) if issue_details else "",
        "issue_codes": ";".join(issue_codes) if issue_codes else "",
        "gape_units": gape_units,
        "frames_tracked": frames_tracked,
        "nan_ratio": round(nan_ratio, 4),
        "gape_std": round(gape_std, 4),
        "max_step": round(max_step, 4),
        "t_force_peak": round(t_force_peak, 4),
        "t_gape_peak": round(t_gape_peak, 4) if np.isfinite(t_gape_peak) else np.nan,
        "peak_diff_s": round(peak_diff, 4) if np.isfinite(peak_diff) else np.nan,
        "t_force_drop": round(t_force_drop, 4) if t_force_drop is not None else np.nan,
        "t_gape_drop": round(t_gape_drop, 4) if t_gape_drop is not None else np.nan,
        "drop_diff_s": round(drop_diff, 4) if np.isfinite(drop_diff) else np.nan,
        "t_mech_start": round(mech_start, 4) if mech_start is not None else np.nan,
        "t_first_gape": round(t_first_gape, 4) if np.isfinite(t_first_gape) else np.nan,
        "start_diff_s": round(start_diff, 4) if np.isfinite(start_diff) else np.nan,
        "last_gape_time": round(last_gape_time, 4) if np.isfinite(last_gape_time) else np.nan,
        "end_gap_s": round(end_gap, 4) if np.isfinite(end_gap) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synchronized CSV outputs.")
    parser.add_argument("--results-dir", type=str, default="results/batch",
                        help="Directory containing synchronized CSV outputs")
    parser.add_argument("--report-path", type=str, default="results/batch/validation_report.csv",
                        help="Output CSV report path")
    parser.add_argument("--exclude-config", type=str, default=None,
                        help="Optional sync override config to exclude samples")
    parser.add_argument("--min-frames", type=int, default=50,
                        help="Minimum tracked frames required (default: 50)")
    parser.add_argument("--max-nan-ratio", type=float, default=0.2,
                        help="Maximum allowed NaN ratio in gape series (default: 0.2)")
    parser.add_argument("--min-gape-std", type=float, default=1.0,
                        help="Minimum standard deviation for gape series (default: 1.0)")
    parser.add_argument("--max-step", type=float, default=0.0,
                        help="Maximum allowed per-step change in gape units (0 = disable)")
    parser.add_argument("--max-peak-diff-s", type=float, default=5.0,
                        help="Maximum allowed peak alignment difference (seconds)")
    parser.add_argument("--max-drop-diff-s", type=float, default=5.0,
                        help="Maximum allowed drop alignment difference (seconds)")
    parser.add_argument("--max-start-diff-s", type=float, default=10.0,
                        help="Maximum allowed start alignment difference (seconds)")
    parser.add_argument("--max-end-gap-s", type=float, default=10.0,
                        help="Maximum allowed gap between last gape point and end time (seconds)")
    parser.add_argument("--drop-smooth-window", type=int, default=5,
                        help="Smoothing window for drop detection (default: 5)")
    parser.add_argument("--force-threshold", type=float, default=0.1,
                        help="Force threshold for mechanical start (default: 0.1)")
    parser.add_argument("--stroke-threshold", type=float, default=0.01,
                        help="Stroke threshold for mechanical start (default: 0.01)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_path = Path(args.report_path)

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    exclude_samples = set()
    if args.exclude_config:
        config_path = Path(args.exclude_config)
        if config_path.exists():
            if config_path.suffix in {".yaml", ".yml"}:
                with config_path.open("r") as f:
                    data = yaml.safe_load(f) or {}
                samples = data.get("samples", {}) or {}
                exclude_samples.update(samples.keys())
            else:
                with config_path.open(newline="") as f:
                    reader = csv.reader(f)
                    for idx, row in enumerate(reader):
                        if not row:
                            continue
                        sample = row[0].strip()
                        if idx == 0 and sample.lower() == "sample":
                            continue
                        if sample.lower() in ("default", "*"):
                            continue
                        if sample:
                            exclude_samples.add(sample)
        else:
            print(f"Warning: Exclude config not found: {config_path}")

    csv_paths = sorted(results_dir.rglob("*_synchronized.csv"))
    if exclude_samples:
        csv_paths = [
            path for path in csv_paths
            if path.stem.replace("_synchronized", "") not in exclude_samples
        ]
    if not csv_paths:
        raise SystemExit(f"No synchronized CSV files found in {results_dir}")

    records: List[Dict[str, object]] = []
    for csv_path in csv_paths:
        try:
            records.append(validate_csv(csv_path, args))
        except Exception as exc:
            records.append({
                "sample": csv_path.stem.replace("_synchronized", ""),
                "path": str(csv_path),
                "status": "FAIL",
                "issues": f"exception: {exc}",
                "issue_codes": "exception",
            })

    report_df = pd.DataFrame(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    total = len(report_df)
    status_counts = report_df["status"].value_counts().to_dict()
    if "issue_codes" in report_df.columns:
        issue_counts = (
            report_df["issue_codes"]
            .fillna("")
            .str.split(";")
            .explode()
            .str.strip()
        )
        issue_counts = issue_counts[issue_counts != ""].value_counts().to_dict()
    else:
        issue_counts = {}

    print("Validation complete")
    print(f"  Report: {report_path}")
    print(f"  Total files: {total}")
    for status in ["PASS", "WARN", "FAIL"]:
        print(f"  {status}: {status_counts.get(status, 0)}")
    if issue_counts:
        print("  Top issues:")
        for code, count in list(issue_counts.items())[:8]:
            print(f"    {code}: {count}")


if __name__ == "__main__":
    main()
