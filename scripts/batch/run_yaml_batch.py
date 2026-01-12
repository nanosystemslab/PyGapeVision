#!/usr/bin/env python
"""
Run batch processing for all samples listed in the YAML config.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def resolve_batch(sample: str) -> str:
    letter = sample[0]
    return f"Batch_{letter}1-{letter}25"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch using YAML config samples.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/processing_config.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/test_yaml_batch",
        help="Output root directory",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive prompts",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text()) or {}
    samples = sorted((config.get("samples") or {}).keys())
    if not samples:
        raise SystemExit("No samples found in config.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    failures = []
    for sample in samples:
        batch = resolve_batch(sample)
        video = Path("data") / "Video" / batch / f"{sample}.mp4"
        csv_path = Path("data") / "Shimadzu" / batch / f"{sample}.csv"
        output_dir = output_root / f"Batch_{sample[0]}" / sample
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "poetry",
            "run",
            "python",
            "scripts/analyze_synced.py",
            str(video),
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ]
        if args.no_interactive:
            cmd.append("--no-interactive")

        print(f"Running {sample}...")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(sample)

    if failures:
        print("Failures:")
        for sample in failures:
            print(f"  - {sample}")
    else:
        print("All samples completed successfully.")


if __name__ == "__main__":
    main()
