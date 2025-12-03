# Manual Tracking Initialization Guide

When automated tracking fails to detect markers, you can manually initialize the tracking points.

## Step 1: Select Tracking Points Interactively

Run the point selection tool on the video:

```bash
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A4.mp4
```

**Instructions:**
1. A window will open showing the first frame
2. Click on the **SHAFT marker** (green paint on hook shaft)
3. Click on the **TIP marker** (green paint on hook tip)
4. Press any key to confirm

The script will output the coordinates:
```
Shaft position: (724, 460)
Tip position:   (902, 707)

Use these positions with:
  --init-shaft-pos 724,460
  --init-tip-pos 902,707
```

## Step 2: Run Analysis with Manual Initialization

Use the coordinates from Step 1:

```bash
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A4.mp4 \
  data/Shimadzu/Batch_A1-A25/A4.csv \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A4 \
  --init-shaft-pos 724,460 \
  --init-tip-pos 902,707
```

## Example: Processing Failed Files

### A4
```bash
# Step 1: Get coordinates
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A4.mp4

# Step 2: Run with manual init (use coordinates from step 1)
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A4.mp4 \
  data/Shimadzu/Batch_A1-A25/A4.csv \
  --frame-skip 10 --fps 30 --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A4 \
  --init-shaft-pos X,Y \
  --init-tip-pos X,Y
```

### A21
```bash
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A21.mp4
# Then run analyze_synced.py with the output coordinates...
```

### A24
```bash
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A24.mp4
# Then run analyze_synced.py with the output coordinates...
```

## How It Works

The manual initialization seeds the temporal tracker with starting positions:
- The tracker searches near these positions in the first frame
- Once locked on, it continues tracking frame-to-frame using proximity search
- This bypasses the automated spatial zone detection which may fail in some videos

## Tips

- Click accurately on the center of each green marker
- The tracker will search nearby for the actual contour centroid
- If tracking still fails, the paint may be too faint or the camera setup different
