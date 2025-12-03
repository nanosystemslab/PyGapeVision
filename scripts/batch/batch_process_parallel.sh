#!/bin/bash
# Parallel batch processing - much faster than sequential
#
# Uses GNU parallel to process multiple videos simultaneously
# Install: brew install parallel (on macOS)
#
# NOTE: Run from project root directory:
#       bash scripts/batch/batch_process_parallel.sh

# Configuration
VIDEO_BASE="data/Video"
CSV_BASE="data/Shimadzu"
OUTPUT_BASE="results/batch"
FPS=30
FRAME_SKIP=10
PIXELS_PER_MM="12.103"

# Parallel processing settings
NUM_JOBS=10  # Number of videos to process simultaneously
             # Adjusted for your 12-core CPU
             # Using 10 cores (leaving 2 for system)

# Skip video output by default for speed (can override below)
# SKIP_VIDEO="--no-video"  # COMMENTED OUT - Videos ENABLED

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "================================================================"
echo "Parallel Batch Processing"
echo "================================================================"
echo "Processing $NUM_JOBS videos simultaneously"
echo "Video output: $([ -n "$SKIP_VIDEO" ] && echo "DISABLED (faster)" || echo "ENABLED (slower)")"
echo "================================================================"
echo ""

# Function to process a single video
# Takes a single argument: "video_file|csv_batch_dir"
process_video() {
    local pair="$1"

    # Split by pipe character
    local video_file="${pair%%|*}"
    local csv_batch_dir="${pair##*|}"

    # Get base name (e.g., A1 from A1.mp4) using parameter expansion
    local filename="${video_file##*/}"  # Remove path
    local base_name="${filename%.mp4}"  # Remove .mp4 extension

    # Extract batch letter (e.g., A from A1)
    local batch_letter="${base_name:0:1}"

    # Check if corresponding CSV exists
    local csv_file="$csv_batch_dir/${base_name}.csv"

    if [ ! -f "$csv_file" ]; then
        echo "⚠️  Skipping $base_name - CSV not found"
        return 1
    fi

    # Create output directory
    local output_dir="$OUTPUT_BASE/Batch_${batch_letter}/$base_name"
    mkdir -p "$output_dir"

    echo "Processing: $base_name"

    # Build command
    local cmd="python scripts/analyze_synced.py \"$video_file\" \"$csv_file\" \
        --frame-skip $FRAME_SKIP \
        --fps $FPS \
        --output-dir \"$output_dir\" \
        --calculate-delta-gape \
        --use-datasheet-initial-gape \
        --exclude-right-pixels 400 \
        $SKIP_VIDEO"

    # Add pixels-per-mm if set
    if [ -n "$PIXELS_PER_MM" ]; then
        cmd="$cmd --pixels-per-mm $PIXELS_PER_MM"
    fi

    # Run analysis
    eval $cmd > "${output_dir}/${base_name}_processing.log" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Check if tracking data was collected
        local csv_file_out="${output_dir}/${base_name}_synchronized.csv"
        if [ -f "$csv_file_out" ]; then
            local data_lines=$(tail -n +2 "$csv_file_out" | wc -l | tr -d ' ')
            if [ "$data_lines" -lt 10 ]; then
                echo "⚠️  $base_name completed but low detection ($data_lines frames)"
                return 2
            else
                echo "✅ $base_name completed ($data_lines frames)"
                return 0
            fi
        else
            echo "⚠️  $base_name completed but no CSV output"
            return 2
        fi
    else
        echo "❌ $base_name failed (exit code $exit_code)"
        return 1
    fi
}

# Export function and variables for parallel
export -f process_video
export VIDEO_BASE CSV_BASE OUTPUT_BASE FPS FRAME_SKIP PIXELS_PER_MM SKIP_VIDEO

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU parallel is not installed"
    echo ""
    echo "Install with:"
    echo "  macOS:   brew install parallel"
    echo "  Linux:   sudo apt-get install parallel"
    echo ""
    echo "Or use the sequential batch_process.sh instead"
    exit 1
fi

# Collect all video/CSV pairs
echo "Scanning for videos..."
pairs_file=$(mktemp)

for batch_dir in "$VIDEO_BASE"/*; do
    if [ ! -d "$batch_dir" ]; then
        continue
    fi

    batch_name=$(basename "$batch_dir")
    csv_batch_dir="$CSV_BASE/$batch_name"

    for video_file in "$batch_dir"/*.mp4; do
        if [ -f "$video_file" ]; then
            echo "$video_file|$csv_batch_dir" >> "$pairs_file"
        fi
    done
done

total_videos=$(wc -l < "$pairs_file" | tr -d ' ')
echo "Found $total_videos videos to process"
echo ""
echo "Starting parallel processing with $NUM_JOBS jobs..."
echo "This will be much faster than sequential processing!"
echo ""

# Process in parallel
# Pass each line as a single argument to process_video
cat "$pairs_file" | parallel -j "$NUM_JOBS" --bar process_video {}

# Clean up temp file
rm "$pairs_file"

# Generate summary report
echo ""
echo "================================================================"
echo "Parallel Processing Complete!"
echo "================================================================"
echo ""
echo "Generating summary report..."

# Count results
SUCCESSFUL=0
FAILED=0
LOW_DETECTION=0

for batch_dir in "$OUTPUT_BASE"/Batch_*; do
    if [ ! -d "$batch_dir" ]; then
        continue
    fi

    for sample_dir in "$batch_dir"/*; do
        if [ ! -d "$sample_dir" ]; then
            continue
        fi

        sample_name=$(basename "$sample_dir")
        csv_file="${sample_dir}/${sample_name}_synchronized.csv"

        if [ -f "$csv_file" ]; then
            data_lines=$(tail -n +2 "$csv_file" | wc -l | tr -d ' ')
            if [ "$data_lines" -lt 10 ]; then
                LOW_DETECTION=$((LOW_DETECTION + 1))
            else
                SUCCESSFUL=$((SUCCESSFUL + 1))
            fi
        else
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "Summary:"
echo "  Total: $total_videos"
echo "  Successful: $SUCCESSFUL"
echo "  Low/No detection: $LOW_DETECTION"
echo "  Failed: $FAILED"
echo ""
echo "Results in: $OUTPUT_BASE/"
echo "================================================================"

# Save summary
REPORT_FILE="$OUTPUT_BASE/parallel_batch_report.txt"
{
    echo "Parallel Batch Processing Report"
    echo "Generated: $(date)"
    echo "Jobs: $NUM_JOBS"
    echo "================================"
    echo ""
    echo "Total: $total_videos"
    echo "Successful: $SUCCESSFUL"
    echo "Low/No detection: $LOW_DETECTION"
    echo "Failed: $FAILED"
} > "$REPORT_FILE"

echo "Report saved: $REPORT_FILE"
