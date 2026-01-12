#!/usr/bin/env python
"""
Interactive reviewer for synchronized analysis plots.

Keys:
  g = good/pass
  b = bad/fail
  s = skip
  q = quit
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV (cv2) is required to run this script.") from exc


def load_reviewed(report_path: Path) -> set[Path]:
    if not report_path.exists():
        return set()
    reviewed = set()
    with report_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("path")
            if path:
                reviewed.add(Path(path))
    return reviewed


def append_row(report_path: Path, row: dict) -> None:
    is_new = not report_path.exists()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample", "path", "status", "timestamp"],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def sample_from_path(path: Path) -> str:
    name = path.stem.replace("_synchronized_analysis", "")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review synchronized analysis plots with keyboard input."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/batch",
        help="Root directory containing *_synchronized_analysis.png files",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="results/batch/plot_review.csv",
        help="CSV report output path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already in the report CSV",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Max display width for the image (default: 1600)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="Max display height for the image (default: 900)",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=None,
        help="Window width (defaults to max-width)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=None,
        help="Window height (defaults to max-height)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_path = Path(args.report_path)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    reviewed = load_reviewed(report_path) if args.resume else set()
    png_paths = sorted(results_dir.rglob("*_synchronized_analysis.png"))
    if reviewed:
        png_paths = [p for p in png_paths if p not in reviewed]

    if not png_paths:
        raise SystemExit("No synchronized analysis plots found to review.")

    for path in png_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        window_width = args.window_width or args.max_width
        window_height = args.window_height or args.max_height

        height, width = image.shape[:2]
        scale = min(args.max_width / width, args.max_height / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interp)

        display = resized
        if window_width > new_width or window_height > new_height:
            canvas = (32 * (cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) * 0)).astype(resized.dtype)
            canvas = cv2.merge([canvas, canvas, canvas])
            canvas = cv2.resize(canvas, (window_width, window_height), interpolation=cv2.INTER_NEAREST)
            y0 = max(0, (window_height - new_height) // 2)
            x0 = max(0, (window_width - new_width) // 2)
            canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
            display = canvas

        help_text = "g=good  b=bad  s=skip  q=quit"
        cv2.putText(
            display,
            help_text,
            (20, window_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        window_name = f"Review: {path}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.imshow(window_name, display)

        key = None
        while key not in (ord("g"), ord("b"), ord("s"), ord("q")):
            key = cv2.waitKey(0) & 0xFF

        cv2.destroyWindow(window_name)

        if key == ord("q"):
            break
        if key == ord("s"):
            continue

        status = "PASS" if key == ord("g") else "FAIL"
        append_row(
            report_path,
            {
                "sample": sample_from_path(path),
                "path": str(path),
                "status": status,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )


if __name__ == "__main__":
    main()
