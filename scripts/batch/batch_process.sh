#!/bin/bash
# Batch process all videos in data/Video/*
#
# Features:
# - Calculates delta gape (change from initial position)
# - Auto-loads measured initial gape from master datasheet
# - Tracks success/failure for each sample
# - Generates detailed processing report
#
# NOTE: This script should be run from the project root directory:
#       bash scripts/batch/batch_process.sh

# Configuration
VIDEO_BASE="data/Video"
CSV_BASE="data/Shimadzu"
OUTPUT_BASE="results/batch"
FPS=30
FRAME_SKIP=10
PIXELS_PER_MM="12.103"  # Calibration: 12.103 pixels per mm
SYNC_METHOD="multi_signature"
SYNC_SEARCH_MIN="-30"
SYNC_SEARCH_MAX="30"
SYNC_SEARCH_STEPS="200"
SIGNATURE_FORCE_WEIGHT="0.7"
SIGNATURE_STROKE_WEIGHT="0.3"
SIGNATURE_SMOOTH_WINDOW="5"
SHOW_TRUE_39MM="--show-true-39mm"
CONFIG_YAML="configs/processing_config.yaml"
CONFIG_YAML_ARG=""
if [ -f "$CONFIG_YAML" ]; then
    CONFIG_YAML_ARG="--config $CONFIG_YAML"
fi
SYNC_CONFIG="configs/sync_overrides.csv"
SYNC_CONFIG_ARG=""
if [ -f "$SYNC_CONFIG" ] && [ -z "$CONFIG_YAML_ARG" ]; then
    SYNC_CONFIG_ARG="--sync-config $SYNC_CONFIG"
fi

# Generate annotated videos
SKIP_VIDEO=""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Initialize tracking arrays
FAILED_FILES=()
LOW_DETECTION_FILES=()
SUCCESSFUL_FILES=()
TOTAL_PROCESSED=0

# Process each batch directory
for batch_dir in "$VIDEO_BASE"/*; do
    if [ ! -d "$batch_dir" ]; then
        continue
    fi

    batch_name=$(basename "$batch_dir")
    csv_batch_dir="$CSV_BASE/$batch_name"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "Processing batch: $batch_name"
    echo "════════════════════════════════════════════════════════"

    # Process each video in this batch
    for video_file in "$batch_dir"/*.mp4; do
        if [ ! -f "$video_file" ]; then
            continue
        fi

        # Get base name (e.g., A1 from A1.mp4)
        base_name=$(basename "$video_file" .mp4)

        # Extract batch letter (e.g., A from A1)
        batch_letter="${base_name:0:1}"

        # Check if corresponding CSV exists
        csv_file="$csv_batch_dir/${base_name}.csv"

        if [ ! -f "$csv_file" ]; then
            echo "⚠️  Skipping $base_name - CSV not found: $csv_file"
            continue
        fi

        # Create output directory for this sample (organized by batch)
        output_dir="$OUTPUT_BASE/Batch_${batch_letter}/$base_name"
        mkdir -p "$output_dir"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing: $base_name"
    echo "Video: $video_file"
    echo "CSV:   $csv_file"
    echo "Output: $output_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build command
    cmd="poetry run python scripts/analyze_synced.py \"$video_file\" \"$csv_file\" \
        --frame-skip $FRAME_SKIP \
        --fps $FPS \
        --output-dir \"$output_dir\" \
        $CONFIG_YAML_ARG \
        $SYNC_CONFIG_ARG \
        --sync-method $SYNC_METHOD \
        --sync-search-min $SYNC_SEARCH_MIN \
        --sync-search-max $SYNC_SEARCH_MAX \
        --sync-search-steps $SYNC_SEARCH_STEPS \
        --signature-force-weight $SIGNATURE_FORCE_WEIGHT \
        --signature-stroke-weight $SIGNATURE_STROKE_WEIGHT \
        --signature-smooth-window $SIGNATURE_SMOOTH_WINDOW \
        --calculate-delta-gape \
        --use-datasheet-initial-gape \
        $SHOW_TRUE_39MM \
        --exclude-right-pixels 400 \
        $SKIP_VIDEO"

    # Add pixels-per-mm if set
    if [ -n "$PIXELS_PER_MM" ]; then
        cmd="$cmd --pixels-per-mm $PIXELS_PER_MM"
    fi

    # Run analysis
    eval $cmd > "${output_dir}/${base_name}_processing.log" 2>&1
    exit_code=$?

        TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))

        if [ $exit_code -eq 0 ]; then
            # Check if tracking data was actually collected
            csv_file_out="${output_dir}/${base_name}_synchronized.csv"
            if [ -f "$csv_file_out" ]; then
                # Count data lines (excluding header)
                data_lines=$(tail -n +2 "$csv_file_out" | wc -l | tr -d ' ')

                if [ "$data_lines" -lt 10 ]; then
                    echo "⚠️  $base_name completed but very low detection ($data_lines frames tracked)"
                    LOW_DETECTION_FILES+=("$base_name (only $data_lines frames)")
                else
                    echo "✅ $base_name completed successfully ($data_lines frames tracked)"
                    SUCCESSFUL_FILES+=("$base_name")
                fi
            else
                echo "⚠️  $base_name completed but no output CSV found"
                LOW_DETECTION_FILES+=("$base_name (no CSV output)")
            fi
        else
            echo "❌ $base_name failed with exit code $exit_code"
            FAILED_FILES+=("$base_name")
        fi
        echo ""
    done
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "Batch Processing Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  Total files processed: $TOTAL_PROCESSED"
echo "  Successful: ${#SUCCESSFUL_FILES[@]}"
echo "  Low/No detection: ${#LOW_DETECTION_FILES[@]}"
echo "  Failed: ${#FAILED_FILES[@]}"
echo ""

# Report failed files
if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo "❌ Files that FAILED processing:"
    for file in "${FAILED_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
fi

# Report low detection files
if [ ${#LOW_DETECTION_FILES[@]} -gt 0 ]; then
    echo "⚠️  Files with LOW or NO marker detection:"
    for file in "${LOW_DETECTION_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
fi

# Save failure report to file
REPORT_FILE="$OUTPUT_BASE/batch_processing_report.txt"
{
    echo "Batch Processing Report"
    echo "Generated: $(date)"
    echo "================================"
    echo ""
    echo "Total files processed: $TOTAL_PROCESSED"
    echo "Successful: ${#SUCCESSFUL_FILES[@]}"
    echo "Low/No detection: ${#LOW_DETECTION_FILES[@]}"
    echo "Failed: ${#FAILED_FILES[@]}"
    echo ""

    if [ ${#FAILED_FILES[@]} -gt 0 ]; then
        echo "FAILED FILES:"
        for file in "${FAILED_FILES[@]}"; do
            echo "  - $file"
        done
        echo ""
    fi

    if [ ${#LOW_DETECTION_FILES[@]} -gt 0 ]; then
        echo "LOW/NO DETECTION FILES:"
        for file in "${LOW_DETECTION_FILES[@]}"; do
            echo "  - $file"
        done
        echo ""
    fi

    if [ ${#SUCCESSFUL_FILES[@]} -gt 0 ]; then
        echo "SUCCESSFUL FILES:"
        for file in "${SUCCESSFUL_FILES[@]}"; do
            echo "  - $file"
        done
    fi
} > "$REPORT_FILE"

echo "Results saved in: $OUTPUT_BASE/"
echo "Processing report: $REPORT_FILE"
echo "════════════════════════════════════════════════════════"
