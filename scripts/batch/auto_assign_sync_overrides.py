#!/usr/bin/env python
"""
Auto-assign sync methods for WARN samples based on validation issues.
"""

import argparse
import csv
from pathlib import Path

import yaml


COLUMNS = [
    "sample",
    "sync_method",
    "drop_smooth_window",
    "sync_search_min",
    "sync_search_max",
    "sync_search_steps",
    "signature_force_weight",
    "signature_stroke_weight",
    "signature_smooth_window",
    "force_threshold",
    "stroke_threshold",
    "video_change_threshold_px",
    "baseline_frames",
    "min_consecutive",
]


def pick_method(issue_codes: set) -> str:
    if "start_misaligned" in issue_codes and not {"peak_misaligned", "drop_misaligned"} & issue_codes:
        return "start_alignment"
    if "peak_misaligned" in issue_codes or "drop_misaligned" in issue_codes:
        return "multi_signature"
    return "drop_alignment"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-assign sync overrides for WARN samples.")
    parser.add_argument(
        "--report-path",
        type=str,
        default="results/batch/validation_report_not_in_config.csv",
        help="Validation report to read WARN samples from",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="configs/sync_overrides_warn.csv",
        help="Output sync overrides CSV path",
    )
    parser.add_argument(
        "--drop-smooth-window",
        type=int,
        default=11,
        help="drop_alignment smoothing window (default: 11)",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path)
    output_path = Path(args.output_path)

    if not report_path.exists():
        raise SystemExit(f"Validation report not found: {report_path}")

    rows = []
    with report_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "WARN":
                continue
            sample = (row.get("sample") or "").strip()
            if not sample:
                continue
            codes = (row.get("issue_codes") or "").split(";")
            codes = {c.strip() for c in codes if c.strip()}
            method = pick_method(codes)
            entry = {col: "" for col in COLUMNS}
            entry["sample"] = sample
            entry["sync_method"] = method
            if method == "drop_alignment":
                entry["drop_smooth_window"] = str(args.drop_smooth_window)
            rows.append(entry)

    rows.sort(key=lambda item: item["sample"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in {".yaml", ".yml"}:
        config = {"defaults": {"sync": {}, "tracking": {}}, "samples": {}}
        if output_path.exists():
            with output_path.open("r") as f:
                config = yaml.safe_load(f) or config
        for row in rows:
            sample = row["sample"]
            sync = {
                "method": row.get("sync_method") or None,
                "drop_smooth_window": int(row["drop_smooth_window"])
                if row.get("drop_smooth_window") else None,
            }
            sync = {k: v for k, v in sync.items() if v is not None}
            config["samples"].setdefault(sample, {})["sync"] = sync
        with output_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} entries to {output_path}")


if __name__ == "__main__":
    main()
