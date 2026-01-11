#!/usr/bin/env python
"""
Generate sync override config from validation results.

Reads validation_report.csv, filters PASS samples, and writes a sync config
based on each sample's results JSON metadata.
"""

import argparse
import csv
import json
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


def load_metadata(results_path: Path) -> dict:
    with results_path.open() as f:
        data = json.load(f)
    return data.get("metadata", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sync override config from PASS samples.")
    parser.add_argument(
        "--report-path",
        type=str,
        default="results/batch/validation_report.csv",
        help="Path to validation report CSV",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="configs/sync_overrides.csv",
        help="Output sync overrides CSV path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append PASS samples not already in the output config",
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
            if row.get("status") != "PASS":
                continue
            sample = row.get("sample")
            csv_path = row.get("path")
            if not sample or not csv_path:
                continue

            results_path = Path(csv_path).with_name(f"{sample}_results.json")
            if not results_path.exists():
                print(f"Warning: Results JSON missing for {sample}: {results_path}")
                continue

            meta = load_metadata(results_path)

            rows.append({
                "sample": sample,
                "sync_method": meta.get("sync_method") or "",
                "drop_smooth_window": meta.get("sync_drop_smooth_window") or "",
                "sync_search_min": meta.get("sync_search_min") or "",
                "sync_search_max": meta.get("sync_search_max") or "",
                "sync_search_steps": meta.get("sync_search_steps") or "",
                "signature_force_weight": meta.get("sync_signature_force_weight") or "",
                "signature_stroke_weight": meta.get("sync_signature_stroke_weight") or "",
                "signature_smooth_window": meta.get("sync_signature_smooth_window") or "",
                "force_threshold": meta.get("sync_force_threshold") or "",
                "stroke_threshold": meta.get("sync_stroke_threshold") or "",
                "video_change_threshold_px": meta.get("sync_video_change_threshold_px") or "",
                "baseline_frames": meta.get("sync_baseline_frames") or "",
                "min_consecutive": meta.get("sync_min_consecutive") or "",
            })

    existing_rows = {}
    if args.append and output_path.exists():
        with output_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = (row.get("sample") or "").strip()
                if sample:
                    existing_rows[sample] = row

    if existing_rows:
        for row in rows:
            sample = row["sample"]
            if sample in existing_rows:
                continue
            existing_rows[sample] = row
        rows = list(existing_rows.values())

    rows.sort(key=lambda item: item.get("sample", ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix in {".yaml", ".yml"}:
        config = {"defaults": {"sync": {}, "tracking": {}}, "samples": {}}
        if args.append and output_path.exists():
            with output_path.open("r") as f:
                config = yaml.safe_load(f) or config
        for row in rows:
            sample = row["sample"]
            sync = {
                "method": row.get("sync_method") or None,
                "drop_smooth_window": row.get("drop_smooth_window") or None,
                "search_min": row.get("sync_search_min") or None,
                "search_max": row.get("sync_search_max") or None,
                "search_steps": row.get("sync_search_steps") or None,
                "signature_force_weight": row.get("signature_force_weight") or None,
                "signature_stroke_weight": row.get("signature_stroke_weight") or None,
                "signature_smooth_window": row.get("signature_smooth_window") or None,
                "force_threshold": row.get("force_threshold") or None,
                "stroke_threshold": row.get("stroke_threshold") or None,
                "video_change_threshold_px": row.get("video_change_threshold_px") or None,
                "baseline_frames": row.get("baseline_frames") or None,
                "min_consecutive": row.get("min_consecutive") or None,
            }
            sync = {k: v for k, v in sync.items() if v not in ("", None)}
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
