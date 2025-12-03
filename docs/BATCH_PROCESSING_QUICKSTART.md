# Batch Processing Quick Start Guide

Complete workflow for processing all videos and populating the master datasheet.

## Prerequisites

1. **Master Datasheet**: `data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv`
   - Must contain measured initial gape values

2. **Video Files**: Organized in `data/Video/Batch_*/`
   - Example: `data/Video/Batch_A1-A25/A1.mp4`

3. **Shimadzu CSV Files**: Organized in `data/Shimadzu/Batch_*/`
   - Example: `data/Shimadzu/Batch_A1-A25/A1.csv`

## Step 1: Run Batch Processing

The batch script has been updated to automatically:
- Calculate delta gape (change from initial position)
- Load measured initial gape from master datasheet
- Track success/failure for each sample

```bash
# Run from project root
bash scripts/batch/batch_process.sh
```

**What it does**:
- Processes all videos in `data/Video/*`
- For each video (e.g., A1.mp4):
  1. Tracks green markers
  2. Syncs with mechanical data
  3. Looks up initial gape (20.9mm) from datasheet
  4. Calculates delta gape = tracked gape - 20.9mm
  5. Generates plots and CSV files
- Saves results to `results/batch/Batch_X/SampleY/`
- Creates processing report: `results/batch/batch_processing_report.txt`

**Output per sample**:
```
results/batch/Batch_A/A1/
├── A1_synchronized.csv              # Includes delta gape columns
├── A1_synchronized_analysis.png     # 4-panel plot with delta gape
├── A1_tracking_analysis.png
├── A1_force_vs_gape.png
├── A1_stroke_vs_force.png
├── A1_results.json                  # Metadata with initial gape info
├── A1_tracked.mp4                   # Annotated video
└── A1_processing.log                # Processing logs
```

## Step 2: Review Failed Samples

Check the processing report:

```bash
cat results/batch/batch_processing_report.txt
```

Expected output:
```
Total files processed: 172
Successful: 169
Low/No detection: 0
Failed: 3

FAILED FILES:
  - A4
  - A21
  - A24
```

## Step 3: Process Failed Samples Manually

For samples that failed automated detection, use manual tracking with HSV override:

### For A4:

```bash
# Step 1: Select tracking points interactively
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A4.mp4
# Click on shaft marker, then tip marker
# Output: --init-shaft-pos 751,450 --init-tip-pos 907,709

# Step 2: Check HSV values (optional, for troubleshooting)
python scripts/debug/check_hsv_at_position.py data/Video/Batch_A1-A25/A4.mp4 751,450
python scripts/debug/check_hsv_at_position.py data/Video/Batch_A1-A25/A4.mp4 907,709

# Step 3: Run with manual tracking and relaxed HSV range
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
  --hsv-upper 60,255,255 \
  --calculate-delta-gape \
  --use-datasheet-initial-gape
```

### Repeat for A21 and A24:

```bash
# A21
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A21.mp4
# ... then run analyze_synced.py with the coordinates

# A24
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A24.mp4
# ... then run analyze_synced.py with the coordinates
```

## Step 4: Populate Master Datasheet

Fill the empty columns in your master datasheet with video analysis results:

```bash
python scripts/batch/populate_master_datasheet.py
```

**Output**:
```
======================================================================
Populating Master Datasheet
======================================================================
Loading master datasheet: data/PIRO_TRT--[...].csv
  Loaded 210 samples
  Using calibration: 12.103 pixels/mm

Processing samples...
  ✓ A1 - Updated
  ✓ A2 - Updated
  ...
  ✓ H25 - Updated

======================================================================
Summary:
  Samples updated: 172
  Samples missing: 0

Updated datasheet saved to:
  data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests_FILLED.csv
======================================================================
```

The filled datasheet now contains:
- **Time to 39mm Gape** (seconds)
- **Force at 39mm Gape** (Newtons)
- **Time to failure** (seconds, at max force)
- **Gape at Failure** (mm, at max force)

## Step 5: Collect Summary Metrics

Generate a summary CSV/Excel with all key metrics:

```bash
python scripts/batch/collect_batch_metrics.py
```

**Output**: `results/batch/batch_metrics_summary.csv` and `.xlsx`

Columns include:
- Sample
- Time_to_39mm_Gape_s
- Force_at_39mm_Gape_N
- Time_to_Failure_s
- Gape_at_Failure_mm
- Initial_Gape_mm (from datasheet)
- Delta_Gape_at_Failure_mm (actual deformation)

## Expected Timeline

For 172 samples on a typical machine:
- **Batch processing**: ~4-6 hours (depending on frame skip and hardware)
- **Manual failed samples**: ~15-30 minutes (3 samples)
- **Populate datasheet**: ~1-2 minutes
- **Collect metrics**: ~1 minute

**Total**: 4-6.5 hours for complete processing

## Troubleshooting

### "Warning: Could not load initial gape for X from datasheet"

**Cause**: Video filename doesn't match datasheet or initial gape is missing.

**Solution**:
1. Check that `A1.mp4` matches "A1" in datasheet "Test #" column
2. Verify initial gape value exists in datasheet
3. Delta gape will fall back to using first tracked frame

### "Error: Master datasheet not found"

**Cause**: Datasheet not at expected location.

**Solution**:
- Ensure file exists at: `data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv`
- Check filename matches exactly (including hyphens and underscores)

### Processing is too slow

**Solutions**:
1. Increase `FRAME_SKIP` in batch_process.sh (e.g., from 10 to 20)
2. Add `--no-video` flag to skip annotated video creation
3. Process in parallel (advanced - requires script modification)

### Many samples fail with "Tracked 0 frames"

**Solutions**:
1. Check HSV range is appropriate: run `python scripts/debug/check_hsv_at_position.py` on a few samples
2. Consider adjusting default HSV range in batch script if needed
3. Review camera setup/lighting consistency

## Verification

After completing all steps, verify:

✓ **Individual results**: Check a few samples in `results/batch/Batch_*/`
  - CSV has delta gape columns
  - Plots show 4 panels (force, stroke, gape, delta gape)
  - JSON metadata shows `"initial_gape_source": "datasheet"`

✓ **Filled datasheet**: Open `data/PIRO_TRT--[...]_FILLED.csv`
  - Empty columns now populated
  - Values look reasonable (no NaN for successful samples)

✓ **Summary metrics**: Open `results/batch/batch_metrics_summary.xlsx`
  - All samples present
  - Initial_Gape_mm values match datasheet
  - Delta_Gape_at_Failure_mm shows actual deformation

## Next Steps

With complete results, you can now:
1. Statistical analysis across hook types/diameters
2. Compare actual deformation (delta gape) vs absolute gape
3. Correlate failure modes with gape behavior
4. Generate publication-ready figures
5. Export to other analysis software

## Configuration Reference

Current batch processing settings (in `scripts/batch/batch_process.sh`):

```bash
FPS=30                    # Video frame rate
FRAME_SKIP=10            # Process every 10th frame
PIXELS_PER_MM="12.103"   # Calibration factor
--calculate-delta-gape   # Enable delta calculation
--use-datasheet-initial-gape  # Auto-load from datasheet
```

To modify:
1. Edit `scripts/batch/batch_process.sh`
2. Change configuration variables at top
3. Re-run batch processing
