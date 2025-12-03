# Master Datasheet Workflow

Complete workflow for using the master datasheet with measured initial gape values.

## Overview

The master datasheet (`data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv`) contains:
- **Known values**: Test metadata (hook type, diameter, etc.) and measured initial gape
- **Empty columns to fill**: Time to 39mm Gape, Force at 39mm Gape, Time to failure, Gape at Failure

This workflow shows how to:
1. Use measured initial gape values during analysis for accurate delta gape calculations
2. Populate the empty columns in the master datasheet with video analysis results

## Why Use Measured Initial Gape?

**Problem**: The first tracked video frame may not capture the true initial gape due to:
- Video starting after test begins
- Tracking detection lag
- Frame skip settings

**Solution**: Use the measured initial gape from the datasheet for delta gape calculation:
```
Delta Gape = Current Gape - Measured Initial Gape
```

This gives you accurate deformation measurements even if tracking starts mid-test.

## Method 1: Automatic from Datasheet (Recommended)

Use the `--use-datasheet-initial-gape` flag to automatically load initial gape based on video filename:

```bash
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A1.mp4 \
  data/Shimadzu/Batch_A1-A25/A1.csv \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A1 \
  --calculate-delta-gape \
  --use-datasheet-initial-gape
```

**How it works**:
- Video filename `A1.mp4` → looks up sample "A1" in datasheet
- Finds initial gape: 20.9 mm
- Calculates: `Delta Gape = Tracked Gape - 20.9mm`

## Method 2: Manual Initial Gape

Provide initial gape explicitly with `--initial-gape`:

```bash
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A1.mp4 \
  data/Shimadzu/Batch_A1-A25/A1.csv \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A1 \
  --calculate-delta-gape \
  --initial-gape 20.9
```

## Method 3: Auto-detect from First Frame (Default)

If neither flag is provided, uses the first tracked frame as initial:

```bash
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A1.mp4 \
  data/Shimadzu/Batch_A1-A25/A1.csv \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A1 \
  --calculate-delta-gape
```

## Batch Processing with Datasheet Initial Gape

Update your batch processing script to use `--use-datasheet-initial-gape`:

```bash
# In batch_process.sh, update the analyze_synced.py command:
python scripts/analyze_synced.py \
  "$video_file" \
  "$csv_file" \
  --frame-skip 10 \
  --fps 30 \
  --pixels-per-mm 12.103 \
  --output-dir "$output_dir" \
  --calculate-delta-gape \
  --use-datasheet-initial-gape
```

## Populating the Master Datasheet

After running batch analysis, populate the empty columns:

```bash
python scripts/batch/populate_master_datasheet.py
```

**What it does**:
1. Reads master datasheet
2. For each sample (A1-H25):
   - Finds corresponding `*_synchronized.csv` in `results/batch/`
   - Extracts metrics:
     - Time to 39mm Gape
     - Force at 39mm Gape
     - Time to failure (at max force)
     - Gape at Failure
3. Fills empty columns in datasheet
4. Saves to: `data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests_FILLED.csv`

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
  Samples updated: 169
  Samples missing: 3

Updated datasheet saved to: data/PIRO_TRT--[...]_FILLED.csv
======================================================================
```

## Complete Workflow Example

### Step 1: Run Batch Analysis with Datasheet Initial Gape

```bash
# Edit batch_process.sh to add --use-datasheet-initial-gape
# Then run batch processing
bash scripts/batch/batch_process.sh
```

### Step 2: Handle Failed Samples

```bash
# Check which samples failed
cat batch_results/batch_processing_report.txt

# For failed samples (e.g., A4), use manual tracking with HSV override
python scripts/calibration/select_tracking_points.py data/Video/Batch_A1-A25/A4.mp4
# Output: --init-shaft-pos 751,450 --init-tip-pos 907,709

python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A4.mp4 \
  data/Shimadzu/Batch_A1-A25/A4.csv \
  --frame-skip 10 --fps 30 --pixels-per-mm 12.103 \
  --output-dir results/batch/Batch_A/A4 \
  --init-shaft-pos 751,450 --init-tip-pos 907,709 \
  --hsv-lower 35,50,50 --hsv-upper 60,255,255 \
  --calculate-delta-gape \
  --use-datasheet-initial-gape
```

### Step 3: Populate Master Datasheet

```bash
python scripts/batch/populate_master_datasheet.py
```

### Step 4: Collect Summary Metrics

```bash
python scripts/batch/collect_batch_metrics.py
```

This creates `results/batch/batch_metrics_summary.csv` with all metrics including delta gape.

## Output Files

**Per-sample outputs** (in `results/batch/Batch_X/SampleY/`):
- `SampleY_synchronized.csv` - Synchronized data with delta gape columns
- `SampleY_synchronized_analysis.png` - 4-panel plot with delta gape
- `SampleY_results.json` - Metadata including initial gape source

**Batch outputs**:
- `results/batch/batch_metrics_summary.csv` - All samples' key metrics
- `results/batch/batch_metrics_summary.xlsx` - Formatted Excel version
- `data/PIRO_TRT--[...]_FILLED.csv` - Master datasheet with filled columns

## Columns in Synchronized CSV

When using `--calculate-delta-gape --use-datasheet-initial-gape`:

| Column | Description |
|--------|-------------|
| `Time` | Mechanical test time (s) |
| `Force` | Tensile force (N) |
| `Stroke` | Crosshead displacement (mm) |
| `Gape_Distance_px` | Absolute gape distance (pixels) |
| `Initial_Gape_px` | Measured initial gape from datasheet (pixels) |
| `Initial_Gape_mm` | Measured initial gape from datasheet (mm) |
| `Delta_Gape_px` | Change in gape from initial (pixels) |
| `Delta_Gape_mm` | Change in gape from initial (mm) - if `--pixels-per-mm` provided |
| `Video_Time_Offset` | Time sync offset (s) |

## Metadata in Results JSON

```json
{
  "metadata": {
    "calculate_delta_gape": true,
    "initial_gape_mm": 20.9,
    "initial_gape_px": 253.154,
    "initial_gape_source": "datasheet",
    "pixels_per_mm": 12.103,
    ...
  }
}
```

**initial_gape_source** values:
- `"datasheet"` - Loaded from master datasheet
- `"manual"` - User provided via `--initial-gape`
- `"first_frame"` - Auto-detected from first tracked frame

## Troubleshooting

### "Warning: Could not load initial gape for X from datasheet"

**Cause**: Sample name in video filename doesn't match datasheet or initial gape is empty.

**Solution**:
1. Check video filename matches Test # column (e.g., `A1.mp4` → `A1`)
2. Verify datasheet has initial gape value for that sample
3. Or use `--initial-gape X.X` to provide manually

### "Warning: Master datasheet not found"

**Cause**: Datasheet not at expected location.

**Solution**:
- Ensure datasheet is at: `data/PIRO_TRT--Additional_Gear_Testing_Master_DataSheet-Tensile_Tests.csv`
- Or update path in `src/datasheet.py`

## Benefits Summary

✓ **Accurate deformation**: Delta gape based on true measured initial values
✓ **Complete dataset**: Master datasheet automatically filled with video results
✓ **Traceability**: Metadata records where initial gape came from
✓ **Flexibility**: Can override with manual values if needed
✓ **Batch-friendly**: Automatically looks up initial gape per sample
