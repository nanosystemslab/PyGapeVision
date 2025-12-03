# Tracking Methods

PyGapeVision uses multiple tracking methods with automatic fallback for robust marker detection.

## Tracking Method Hierarchy

The system tries methods in this order:

1. **Spatial Zones** (default) - Divides frame into regions based on expected hook orientation
2. **Manual Initialization** - User provides starting positions for shaft and tip markers
3. **Simple Tracking** (automatic fallback) - Tracks the two largest green contours

## Method Details

### 1. Spatial Zones (`tracking_method: "spatial_zones"`)

**How it works**:
- Divides the frame into spatial zones based on typical hook orientation
- Shaft zone: upper-middle area (y: 380-520, x: 620-850)
- Tip zone: lower-right area (y: 650-800, x: 800-1000)
- Validates tip has two distinct edges (left and right sides of hook point)
- Uses temporal tracking: once locked on, follows markers frame-to-frame

**Advantages**:
- Most accurate - uses hook geometry knowledge
- Validates tip using edge detection
- Robust to lighting variations

**When it works best**:
- Consistent camera setup
- Hook in expected orientation
- Clear green markers

### 2. Manual Initialization (`tracking_method: "manual_init"`)

**How it works**:
- User clicks on shaft and tip positions in first frame
- System searches near those positions for green markers
- Continues with temporal tracking from there

**Advantages**:
- Works when hook is in unusual orientation
- Handles different camera angles
- User controls which points to track

**How to use**:
```bash
# Step 1: Select points interactively
python scripts/calibration/select_tracking_points.py video.mp4

# Step 2: Use coordinates with analysis
python scripts/analyze_synced.py video.mp4 data.csv \
  --init-shaft-pos 751,450 \
  --init-tip-pos 907,709
```

**When to use**:
- Spatial zones method fails
- Non-standard camera setup
- Hook in unusual orientation

### 3. Simple Tracking (`tracking_method: "simple"`)

**How it works**:
- Finds all green regions in frame
- Selects two largest contours
- Determines shaft vs tip based on y-coordinate (upper = shaft, lower = tip)
- No spatial constraints or edge validation

**Advantages**:
- Very robust - works if there are ANY two green points
- No assumptions about hook orientation
- Automatic fallback - no user intervention needed

**When it activates**:
- Automatically if other methods track < 10 frames
- System retries automatically and compares results

**Limitations**:
- Less accurate - doesn't validate tip geometry
- May pick wrong points if >2 green regions exist
- No edge validation

## Automatic Fallback System

The system automatically tries fallback methods:

```
1. Try spatial zones method
   └─> < 10 frames tracked?

2. Retry with simple tracking method
   └─> More frames tracked with simple method?
       ├─> YES: Use simple method results
       └─> NO:  Keep original results (may have manual init)
```

**Example output**:
```
[2/4] Processing video for synchronization...
  First pass: Tracking points...
  Processing complete. Tracked 3 frames.
  ⚠️  Warning: Only 3 frames tracked with spatial_zones method
  Retrying with simple tracking method (two largest contours)...
  Processing complete. Tracked 145 frames.
  ✓ Simple tracking method succeeded: 145 frames tracked
```

## Tracking Method in Results

The method used is recorded in the output JSON:

```json
{
  "metadata": {
    "tracking_method": "simple",
    "frames_tracked": 145,
    ...
  }
}
```

**Tracking method values**:
- `"spatial_zones"` - Default zone-based detection
- `"manual_init"` - User provided initial positions
- `"simple"` - Two largest contours fallback method

## Choosing the Right Method

### Use Spatial Zones When:
- ✓ Standard camera setup
- ✓ Hook in typical orientation
- ✓ Clear, bright green markers
- ✓ Batch processing with consistent setup

### Use Manual Initialization When:
- ✓ Spatial zones fails (<10 frames)
- ✓ Non-standard camera angle
- ✓ Hook in unusual orientation
- ✓ Simple fallback also fails

### Simple Tracking Activates When:
- ✓ Automatically if <10 frames tracked
- ✓ Tries automatically, no user action needed
- ✓ Works if any two green points visible

## HSV Range Override

All methods benefit from correct HSV color range:

```bash
# Default
--hsv-lower 35,100,50
--hsv-upper 55,255,255

# Relaxed for difficult videos
--hsv-lower 35,50,50
--hsv-upper 60,255,255
```

See `docs/HSV_OVERRIDE_GUIDE.md` for details.

## Troubleshooting

### "Tracked 0 frames" with all methods

**Possible causes**:
1. HSV range doesn't match marker color
2. Markers too faint/not visible
3. Wrong video file

**Solutions**:
```bash
# Check HSV at marker positions
python scripts/debug/check_hsv_at_position.py video.mp4 x,y

# Adjust HSV range based on diagnostic output
python scripts/analyze_synced.py ... --hsv-lower 35,40,40 --hsv-upper 70,255,255
```

### Simple tracking picks wrong points

**Symptoms**: Tracks two green points that aren't shaft and tip

**Solutions**:
1. Use manual initialization instead:
   ```bash
   python scripts/calibration/select_tracking_points.py video.mp4
   python scripts/analyze_synced.py ... --init-shaft-pos X,Y --init-tip-pos X,Y
   ```

2. Adjust HSV range to filter out unwanted green regions

### Inconsistent tracking across batch

**Check tracking methods used**:
```bash
# Find which method each sample used
for f in results/batch/Batch_*/*/*.json; do
  echo "$f: $(jq -r '.metadata.tracking_method' $f)"
done
```

**If many use simple method**:
- Consider adjusting default HSV range
- Check camera setup consistency
- Review spatial zone boundaries for your setup

## Best Practices

1. **Batch Processing**: Let automatic fallback handle edge cases
2. **Single Samples**: Check tracking_method in JSON to verify correct detection
3. **Failed Samples**: Use manual initialization with HSV override
4. **Camera Setup**: Consistent setup = more samples use spatial_zones (most accurate)
5. **Validation**: Review a few videos to ensure correct points are tracked

## Method Accuracy Comparison

| Method | Accuracy | Robustness | User Effort |
|--------|----------|------------|-------------|
| Spatial Zones | ★★★★★ | ★★★☆☆ | None |
| Manual Init | ★★★★☆ | ★★★★☆ | Manual selection |
| Simple | ★★★☆☆ | ★★★★★ | None (automatic) |

**Recommendation**: Let the system try fallbacks automatically. Review results and manually process only samples that still fail.
