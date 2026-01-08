# Docker Usage

This image installs PyGapeVision and its runtime dependencies, then runs scripts against data mounted from the host.

## Build

```bash
docker build -t pygapevision:latest .
```

## Run (single video)

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  pygapevision:latest \
  python scripts/analyze_synced.py \
    data/Video/Batch_A1-A25/A1.mp4 \
    data/Shimadzu/Batch_A1-A25/A1.csv \
    --fps 30 \
    --frame-skip 10 \
    --pixels-per-mm 12.103 \
    --calculate-delta-gape \
    --use-datasheet-initial-gape \
    --output-dir results
```

## Run (batch processing)

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  pygapevision:latest \
  bash scripts/batch/batch_process_parallel.sh
```

## Notes

- Bind mounts keep large videos on the host and avoid copying them into the image.
- Use `--rm` to avoid accumulating container layers with large outputs.
- Interactive calibration/debug tools that open windows are not expected to work in a headless container.
