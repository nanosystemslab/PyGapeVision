# Delta Gape Calculation Guide

## Overview

The `--calculate-delta-gape` flag enables calculation of **delta gape** - the change in gape distance from the initial position. This addresses the issue that absolute gape measurements vary based on where paint markers were applied, but the **change in gape** during the test represents the actual deformation.

## Why Use Delta Gape?

**Problem**: Not the exact same physical point is tracked across different hooks due to variations in paint placement.

**Solution**: Track the change from the initial position. Since the same point IS tracked consistently throughout a single test, delta gape gives you the actual deformation:

```
Delta Gape = Current Gape - Initial Gape
```

## Usage

Add the `--calculate-delta-gape` flag to any analysis:

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

## What Gets Calculated

### In Synchronized CSV

Three new columns are added:

1. **Initial_Gape_px**: The gape distance at the first tracked frame (constant for entire test)
2. **Delta_Gape_px**: The change in gape from initial position (`Gape_Distance_px - Initial_Gape_px`)
3. Converted to mm if `--pixels-per-mm` is provided

### In Plots

When delta gape is enabled, the synchronized analysis plot includes a 4th panel:
- **Plot 4**: Delta Gape vs Time with Force overlay
- Shows the gape change from initial position in purple
- Useful for comparing actual deformation across different hooks

### In Batch Metrics

When `collect_batch_metrics.py` detects delta gape columns, it adds:
- **Initial_Gape_mm**: Initial gape distance for the sample
- **Delta_Gape_at_Failure_mm**: Change in gape at maximum force

## Example Scenarios

### Scenario 1: Comparing Hook Deformation

You have two hooks with paint applied at different positions:
- Hook A: Paint at 35mm initial gape
- Hook B: Paint at 40mm initial gape

**Without delta gape**:
- Hook A fails at 45mm absolute gape
- Hook B fails at 50mm absolute gape
- Comparison is misleading!

**With delta gape**:
- Hook A: Delta = 45 - 35 = 10mm deformation
- Hook B: Delta = 50 - 40 = 10mm deformation
- Both hooks deformed the same amount! ✓

### Scenario 2: Batch Processing

Process entire batch with delta gape:

```bash
# Enable delta gape in batch script (edit batch_process.sh)
# Add --calculate-delta-gape to the analyze_synced.py command

# Then collect metrics
python scripts/batch/collect_batch_metrics.py
```

The output CSV will include both absolute and delta gape metrics.

## Metadata Storage

The delta gape calculation is recorded in the results JSON:

```json
{
  "metadata": {
    "calculate_delta_gape": true,
    "initial_gape_px": 324.56,
    "pixels_per_mm": 12.103,
    ...
  }
}
```

## When to Use Delta Gape

**Use delta gape when**:
- Comparing deformation across multiple hooks
- Paint markers are applied at slightly different positions
- You want to measure actual material/structure deformation

**Use absolute gape when**:
- You need the actual physical gape opening
- Paint is consistently applied at the same position across all samples
- You're measuring a specific gape threshold (e.g., "reaches 39mm")

## Combining Both Metrics

You can analyze both:
- Absolute gape tells you when a specific opening size is reached
- Delta gape tells you how much the hook has deformed

Example: "Hook opened by 12mm (delta) and reached 39mm absolute gape"
