#!/usr/bin/env python
"""
Convert sync overrides CSV to unified YAML processing config.
"""

import argparse
import csv
from pathlib import Path

import yaml


def to_number(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sync overrides CSV to YAML config.")
    parser.add_argument("--input-path", type=str, default="configs/sync_overrides.csv",
                        help="Input CSV path")
    parser.add_argument("--output-path", type=str, default="configs/processing_config.yaml",
                        help="Output YAML path")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    config = {"defaults": {"sync": {}, "tracking": {}}, "samples": {}}

    if output_path.exists():
        with output_path.open("r") as f:
            existing = yaml.safe_load(f) or {}
        config["defaults"] = existing.get("defaults", config["defaults"])
        config["samples"] = existing.get("samples", config["samples"])

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = (row.get("sample") or "").strip()
            if not sample:
                continue
            sync = {
                "method": row.get("sync_method") or None,
                "drop_smooth_window": to_number(row.get("drop_smooth_window")),
                "search_min": to_number(row.get("sync_search_min")),
                "search_max": to_number(row.get("sync_search_max")),
                "search_steps": to_number(row.get("sync_search_steps")),
                "signature_force_weight": to_number(row.get("signature_force_weight")),
                "signature_stroke_weight": to_number(row.get("signature_stroke_weight")),
                "signature_smooth_window": to_number(row.get("signature_smooth_window")),
                "force_threshold": to_number(row.get("force_threshold")),
                "stroke_threshold": to_number(row.get("stroke_threshold")),
                "video_change_threshold_px": to_number(row.get("video_change_threshold_px")),
                "baseline_frames": to_number(row.get("baseline_frames")),
                "min_consecutive": to_number(row.get("min_consecutive")),
            }

            sync = {k: v for k, v in sync.items() if v is not None and v != ""}
            config["samples"].setdefault(sample, {})["sync"] = sync

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote YAML config to {output_path}")


if __name__ == "__main__":
    main()
