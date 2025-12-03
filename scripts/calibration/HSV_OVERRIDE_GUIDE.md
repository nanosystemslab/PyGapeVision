# HSV Override Guide

When automated tracking fails due to HSV detection issues, you can override the default HSV range.

## Default HSV Range

```
--hsv-lower 35,100,50
--hsv-upper 55,255,255
```

This targets vibrant yellowish-green markers and excludes:
- Bluer greens (Kryptonite Green at H≈65)
- Gray/desaturated colors (min S=100)

## Troubleshooting HSV Issues

### Step 1: Diagnose the Problem

Use the HSV diagnostic tool to check values at your marker positions:

```bash
python scripts/debug/check_hsv_at_position.py <video_path> <x>,<y>
```

Example for A4:
```bash
python scripts/debug/check_hsv_at_position.py data/Video/Batch_A1-A25/A4.mp4 907,709
```

This will show:
- Exact HSV values at the pixel
- Region average with standard deviation
- Whether it falls within current detection range
- Suggested HSV range for the region

### Step 2: Adjust HSV Range

Common adjustments:

**For markers with lower saturation** (mixing with metal/silver):
```bash
--hsv-lower 35,50,50   # Reduce S minimum from 100 to 50
```

**For markers with slightly different hue**:
```bash
--hsv-upper 60,255,255  # Increase H maximum from 55 to 60
```

**For very faint markers**:
```bash
--hsv-lower 35,40,40   # Further reduce S and V minimums
```

## Processing Failed Files (A4, A21, A24)

Based on diagnostic output showing H:56.1 and S:55.2 in tip region:

### A4 Example

```bash
# Step 1: Get manual positions
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A4.mp4

# Step 2: Run with relaxed HSV range
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A4.mp4 \
  data/Shimadzu/Batch_A1-A25/A4.csv \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A4 \
  --init-shaft-pos 751,450 \
  --init-tip-pos 907,709 \
  --hsv-lower 35,50,50 \
  --hsv-upper 60,255,255
```

### A21 and A24

Follow the same process:

1. Use `select_tracking_points.py` to get coordinates
2. Use `check_hsv_at_position.py` to diagnose HSV issues
3. Run `analyze_synced.py` with adjusted HSV range

## Tips

- Start with small adjustments (±5 for H, ±20 for S/V)
- The diagnostic tool suggests a range based on ±2 standard deviations
- Too wide a range may pick up false positives
- HSV values are saved to results JSON for reference

## HSV Color Space Reference

- **H (Hue)**: 0-179 in OpenCV (color, e.g., 35-55 for yellowish-green)
- **S (Saturation)**: 0-255 (color intensity, 0=gray, 255=vibrant)
- **V (Value)**: 0-255 (brightness, 0=black, 255=bright)
