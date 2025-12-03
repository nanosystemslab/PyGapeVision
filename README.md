# PyGapeVision

Automated video analysis system for tracking hook gape deformation during mechanical tension tests. Synchronizes video tracking with force/displacement data from Shimadzu mechanical testing equipment.

## Overview

PyGapeVision tracks the distance between two green-painted markers on fishing hooks as they deform under load. The system:
- Automatically detects green markers using computer vision
- Tracks marker positions frame-by-frame through video
- Synchronizes video timestamps with mechanical test data (force, displacement, time)
- Calculates both absolute gape and delta gape (change from initial position)
- Generates comprehensive visualizations and data exports
- Supports batch processing of hundreds of samples

## Key Features

- **Automatic Green Marker Detection**: HSV-based color segmentation with configurable thresholds
- **Video-Mechanical Data Synchronization**: Peak alignment algorithm for automatic time offset detection
- **Delta Gape Calculation**: Tracks change from measured initial gape position
- **Batch Processing**: Parallel processing of multiple videos with automatic datasheet population
- **Comprehensive Visualization**: Force vs gape, stroke vs force, time series plots, and annotated videos
- **Calibration Tools**: Pixel-to-mm conversion with interactive calibration interface
- **Manual Tracking Fallback**: For challenging videos with poor automatic detection
- **Ruler Exclusion**: Automatic masking of right edge to avoid interference from in-frame rulers

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/PyGapeVision.git
cd PyGapeVision
```

### 2. Set Up Python Environment

Using pyenv (recommended):

```bash
pyenv virtualenv 3.12.4 pygapevision
pyenv local pygapevision
```

Or using venv:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

For Excel export support:

```bash
pip install -e ".[excel]"
```

## Quick Start

### Single Video Analysis

Analyze a single video with synchronized mechanical data:

```bash
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

This generates:
- `A1_synchronized.csv` - Frame-by-frame data with force, stroke, gape, and delta gape
- `A1_synchronized_analysis.png` - 4-panel plot (force, stroke, gape, delta gape vs time)
- `A1_force_vs_gape.png` - Force-displacement curve
- `A1_stroke_vs_force.png` - Stroke-force relationship
- `A1_tracked.mp4` - Annotated video with tracking overlays
- `A1_results.json` - Metadata and tracking data

### Batch Processing (Recommended)

Process all videos in `data/Video/` with parallel execution:

```bash
# Install GNU parallel first (macOS)
brew install parallel

# Run batch processing (10 videos at once)
bash scripts/batch/batch_process_parallel.sh
```

For sequential processing (no parallel dependency):

```bash
bash scripts/batch/batch_process.sh
```

After processing completes:

```bash
# Generate summary metrics CSV/Excel
python scripts/batch/collect_batch_metrics.py

# Populate master datasheet with results
python scripts/batch/populate_master_datasheet.py
```

See [BATCH_PROCESSING_QUICKSTART.md](BATCH_PROCESSING_QUICKSTART.md) for detailed workflow.

## Project Structure

```
PyGapeVision/
├── data/
│   ├── Video/                          # Input videos organized by batch
│   │   ├── Batch_A1-A25/
│   │   ├── Batch_B1-B25/
│   │   └── ...
│   ├── Shimadzu/                       # Mechanical test CSV files
│   │   ├── Batch_A1-A25/
│   │   └── ...
│   └── PIRO_TRT--...--Master_DataSheet-Tensile_Tests.csv
│
├── src/                                # Core library modules
│   ├── tracker.py                      # Green marker detection & tracking
│   ├── visualization.py                # Plotting functions
│   ├── sync.py                         # Video-mechanical data synchronization
│   ├── calibration.py                  # Pixel-to-mm calibration
│   └── datasheet.py                    # Master datasheet I/O
│
├── scripts/                            # Command-line analysis tools
│   ├── analyze_synced.py               # Main synchronized analysis script
│   ├── batch/
│   │   ├── batch_process_parallel.sh   # Parallel batch processing
│   │   ├── batch_process.sh            # Sequential batch processing
│   │   ├── collect_batch_metrics.py    # Generate summary CSV/Excel
│   │   └── populate_master_datasheet.py # Fill datasheet with results
│   ├── calibration/
│   │   ├── calibrate.py                # Interactive pixel-to-mm calibration
│   │   └── select_tracking_points.py   # Manual tracking point selection
│   └── debug/
│       ├── tune_hsv.py                 # HSV threshold tuning tool
│       └── check_hsv_at_position.py    # Check HSV at specific coordinates
│
├── results/
│   ├── batch/                          # Batch processing outputs
│   │   ├── Batch_A/
│   │   ├── Batch_B/
│   │   ├── batch_metrics_summary.csv   # Summary of all samples
│   │   └── batch_metrics_summary.xlsx
│   └── calibration/                    # Calibration reference files
│
├── docs/                               # Additional documentation
├── tests/                              # Unit tests
├── pyproject.toml                      # Package configuration
├── LICENSE
└── README.md
```

## Usage

### 1. Calibration (One-time Setup)

Calibrate pixel-to-mm conversion using a ruler or known reference:

```bash
python scripts/calibration/calibrate.py data/Video/Batch_A1-A25/A1.mp4 10.0
```

This:
1. Displays first frame
2. Lets you click two points on a 10mm ruler
3. Calculates pixels per mm
4. Saves to `results/calibration/calibration.json`

Current calibration: **12.103 pixels/mm**

### 2. Single Video Analysis

#### Basic Usage

```bash
python scripts/analyze_synced.py VIDEO.mp4 CSV_FILE.csv
```

#### With All Options

```bash
python scripts/analyze_synced.py \
  data/Video/Batch_A1-A25/A1.mp4 \
  data/Shimadzu/Batch_A1-A25/A1.csv \
  --fps 30 \
  --frame-skip 10 \
  --pixels-per-mm 12.103 \
  --calculate-delta-gape \
  --use-datasheet-initial-gape \
  --exclude-right-pixels 400 \
  --output-dir results
```

**Key Parameters**:
- `--fps`: Video frame rate (default: 100)
- `--frame-skip`: Process every Nth frame for speed (default: 5)
- `--pixels-per-mm`: Calibration factor (optional)
- `--calculate-delta-gape`: Calculate change from initial position
- `--use-datasheet-initial-gape`: Load initial gape from master datasheet
- `--exclude-right-pixels`: Mask right edge to avoid ruler interference (default: 400)
- `--no-video`: Skip video output generation (faster)

#### Manual Tracking (For Failed Samples)

When automatic detection fails, use manual initialization:

```bash
# Step 1: Select tracking points interactively
python scripts/calibration/select_tracking_points.py VIDEO.mp4
# Click on shaft marker, then tip marker
# Outputs: --init-shaft-pos X,Y --init-tip-pos X,Y

# Step 2: Run analysis with manual positions
python scripts/analyze_synced.py VIDEO.mp4 CSV_FILE.csv \
  --init-shaft-pos 724,460 \
  --init-tip-pos 902,707 \
  --hsv-lower 35,50,50 \
  --hsv-upper 60,255,255
```

### 3. Batch Processing

See [BATCH_PROCESSING_QUICKSTART.md](BATCH_PROCESSING_QUICKSTART.md) for complete workflow.

#### Quick Summary

```bash
# 1. Process all videos (parallel, 10 jobs)
bash scripts/batch/batch_process_parallel.sh

# 2. Check progress
bash scripts/batch/check_progress.sh

# 3. Collect metrics
python scripts/batch/collect_batch_metrics.py

# 4. Populate master datasheet
python scripts/batch/populate_master_datasheet.py
```

## Output Files

### Per-Sample Outputs

Each analyzed sample generates:

```
results/batch/Batch_A/A1/
├── A1_synchronized.csv              # Frame-by-frame synchronized data
├── A1_synchronized_analysis.png     # 4-panel time series plot
├── A1_force_vs_gape.png            # Force-displacement curve
├── A1_stroke_vs_force.png          # Stroke-force curve
├── A1_tracking_analysis.png        # Tracking quality visualization
├── A1_tracked.mp4                  # Annotated video with overlays
├── A1_results.json                 # Metadata and raw tracking data
└── A1_processing.log               # Processing logs
```

### Synchronized CSV Format

`A1_synchronized.csv` contains:

| Column | Description |
|--------|-------------|
| `Time` | Synchronized time (seconds) |
| `Force` | Force from mechanical test (N) |
| `Stroke` | Displacement from mechanical test (mm) |
| `Gape_Distance_px` | Absolute gape distance (pixels) |
| `Gape_Distance_mm` | Absolute gape distance (mm, if calibrated) |
| `Delta_Gape_px` | Change from initial gape (pixels) |
| `Delta_Gape_mm` | Change from initial gape (mm, if calibrated) |
| `Initial_Gape_px` | Initial gape measurement (pixels) |
| `Initial_Gape_mm` | Initial gape measurement (mm, from datasheet) |
| `Shaft_X`, `Shaft_Y` | Shaft marker position |
| `Tip_X`, `Tip_Y` | Tip marker position |

### Batch Summary Files

After batch processing:

```
results/batch/
├── batch_metrics_summary.csv        # All samples, key metrics
├── batch_metrics_summary.xlsx       # Excel version
└── parallel_batch_report.txt        # Processing summary
```

**Metrics Summary Columns**:
- Sample
- Time_to_39mm_Gape_s
- Force_at_39mm_Gape_N
- Time_to_Failure_s
- Force_at_Failure_N (max force)
- Gape_at_Failure_mm
- Delta_Gape_at_Failure_mm
- Initial_Gape_mm

## Tracking Algorithm

### Green Marker Detection

1. **Color Segmentation**: Convert frame to HSV color space
2. **Thresholding**: Apply HSV range filter (default: H=35-55, S=100-255, V=50-255)
3. **Edge Exclusion**: Mask right edge (400px) to avoid ruler interference
4. **Morphological Ops**: Remove noise and close gaps
5. **Contour Detection**: Find connected green regions

### Tracking Methods

**Method 1: Spatial Zones (Default)**
- Divides frame into left/right zones
- Identifies shaft (left) vs tip (right) based on x-coordinate
- Robust to occlusion and movement

**Method 2: Two Largest Contours (Fallback)**
- Finds two largest green regions
- Assigns based on leftmost = shaft, rightmost = tip
- Used when spatial zones fail

**Method 3: Manual Initialization**
- User provides initial shaft and tip positions
- Tracks nearest green region to last known position
- Best for challenging videos

### Synchronization Algorithm

**Peak Alignment Method**:
1. Calculate gape distance rate of change (video)
2. Calculate force rate of change (mechanical data)
3. Cross-correlate signals to find optimal time offset
4. Align video timeline to mechanical timeline
5. Interpolate to match timestamps

## Troubleshooting

### Green Markers Not Detected

**Symptoms**: "Tracked 0 frames" or very low frame count

**Solutions**:
1. Check HSV range with debugging tool:
   ```bash
   python scripts/debug/check_hsv_at_position.py VIDEO.mp4 X,Y
   ```
2. Adjust HSV thresholds:
   ```bash
   --hsv-lower 30,50,50 --hsv-upper 70,255,255
   ```
3. Use manual tracking initialization (see above)
4. Verify green paint is bright and saturated

### Tracking Jumps or Switches Points

**Cause**: Similar-looking green regions confusing the tracker

**Solutions**:
1. Increase `--exclude-right-pixels` to mask more of the frame
2. Use manual tracking initialization
3. Adjust HSV range to be more restrictive
4. Check for reflections or other green objects in frame

### Ruler Interference

**Symptoms**: Tracking jumps to yellow ruler on right side of frame

**Solution**: Increase exclusion zone (already default 400px):
```bash
--exclude-right-pixels 500
```

### Synchronization Issues

**Symptoms**: Force and gape plots don't align properly

**Solutions**:
1. Check that video starts before mechanical test begins
2. Verify FPS is correct (--fps parameter)
3. Try manual time offset:
   ```bash
   --time-offset 2.5  # Video is 2.5 seconds ahead of mechanical data
   ```
4. Review sync correlation in output (should be > 0.7)

### Performance Issues

**Solutions**:
1. Increase frame skip: `--frame-skip 20` (process fewer frames)
2. Skip video output: `--no-video` (much faster)
3. Use parallel batch processing instead of sequential
4. Process on machine with more CPU cores

## Advanced Usage

### Python API

```python
from src.tracker import GreenPointTracker, VideoAnalyzer
from src.visualization import plot_tracking_results, plot_force_vs_gape
from src.sync import load_shimadzu_csv, auto_sync_video_to_mechanical, sync_data

# Initialize tracker with custom HSV range
tracker = GreenPointTracker(
    hsv_lower=(35, 100, 50),
    hsv_upper=(55, 255, 255),
    exclude_right_pixels=400
)

# Analyze video
analyzer = VideoAnalyzer('video.mp4', tracker)
results = analyzer.process_video(
    output_video_path='output.mp4',
    frame_skip=10,
    fps_override=30
)

# Load mechanical data
shimadzu_df = load_shimadzu_csv('mechanical_data.csv')

# Auto-sync
time_offset, correlation = auto_sync_video_to_mechanical(
    results,
    shimadzu_df,
    method='peak_alignment'
)

# Create synchronized dataset
synced_df = sync_data(
    results,
    shimadzu_df,
    time_offset,
    calculate_delta=True,
    pixels_per_mm=12.103
)

# Generate plots
plot_force_vs_gape(synced_df, output_path='force_gape.png', pixels_per_mm=12.103)
```

### Custom Batch Processing

Modify `scripts/batch/batch_process_parallel.sh` configuration:

```bash
FPS=30                    # Video frame rate
FRAME_SKIP=10            # Process every Nth frame
PIXELS_PER_MM="12.103"   # Calibration factor
NUM_JOBS=10              # Parallel jobs (cores to use)
SKIP_VIDEO="--no-video"  # Uncomment to disable video output
```

## Data Format

### JSON Results Format

```json
{
  "metadata": {
    "video_path": "data/Video/Batch_A1-A25/A1.mp4",
    "video_fps": 30,
    "pixels_per_mm": 12.103,
    "time_offset_seconds": 2.45,
    "correlation": 0.89,
    "initial_gape_mm": 20.9,
    "tracking_method": "spatial_zones",
    "frames_tracked": 1234
  },
  "tracking_data": {
    "frame_numbers": [0, 5, 10, ...],
    "time_seconds": [0.0, 0.167, 0.333, ...],
    "shaft_x": [724, 724, 725, ...],
    "shaft_y": [460, 461, 462, ...],
    "tip_x": [902, 903, 905, ...],
    "tip_y": [707, 708, 710, ...],
    "distance_pixels": [240.5, 241.2, 242.8, ...]
  }
}
```

## Configuration

### HSV Color Ranges

Default green detection:
- **Hue**: 35-55 (green range in HSV)
- **Saturation**: 100-255 (high saturation)
- **Value**: 50-255 (medium to bright)

Adjust for different paint colors or lighting:
```bash
--hsv-lower H,S,V --hsv-upper H,S,V
```

### Calibration

Current system calibration: **12.103 pixels per mm**

To recalibrate:
```bash
python scripts/calibration/calibrate.py VIDEO.mp4 KNOWN_DISTANCE_MM
```

## Testing

Run unit tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

Additional guides in `docs/`:
- [BATCH_PROCESSING_QUICKSTART.md](BATCH_PROCESSING_QUICKSTART.md) - Complete batch workflow
- `scripts/calibration/HSV_OVERRIDE_GUIDE.md` - HSV tuning guide
- `scripts/calibration/MANUAL_TRACKING_GUIDE.md` - Manual tracking guide

## Citation

If you use PyGapeVision in your research, please cite:

```
Nakamura, M. (2024). PyGapeVision: Automated hook gape analysis from video tracking.
GitHub repository: https://github.com/yourusername/PyGapeVision
```

## License

GNU General Public License v3.0 or later (GPLv3+) - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: matthew.t.nakamura@gmail.com

## Acknowledgments

Developed for the Pelagic and Reef Investigations Lab (PIRO) at the University of Hawaii.
