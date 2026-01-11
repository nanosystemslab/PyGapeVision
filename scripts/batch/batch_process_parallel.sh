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
FPS="${FPS:-30}"
FRAME_SKIP="${FRAME_SKIP:-10}"
PIXELS_PER_MM="${PIXELS_PER_MM:-12.103}"
SYNC_METHOD="${SYNC_METHOD:-multi_signature}"
SYNC_SEARCH_MIN="${SYNC_SEARCH_MIN:--30}"
SYNC_SEARCH_MAX="${SYNC_SEARCH_MAX:-30}"
SYNC_SEARCH_STEPS="${SYNC_SEARCH_STEPS:-200}"
SIGNATURE_FORCE_WEIGHT="${SIGNATURE_FORCE_WEIGHT:-0.7}"
SIGNATURE_STROKE_WEIGHT="${SIGNATURE_STROKE_WEIGHT:-0.3}"
SIGNATURE_SMOOTH_WINDOW="${SIGNATURE_SMOOTH_WINDOW:-5}"
SHOW_TRUE_39MM="${SHOW_TRUE_39MM:---show-true-39mm}"
CONFIG_YAML="${CONFIG_YAML:-configs/processing_config.yaml}"
CONFIG_YAML_ARG=""
if [ -f "$CONFIG_YAML" ]; then
    CONFIG_YAML_ARG="--config $CONFIG_YAML"
fi
SYNC_CONFIG="${SYNC_CONFIG:-configs/sync_overrides.csv}"
SYNC_CONFIG_ARG=""
if [ -f "$SYNC_CONFIG" ] && [ -z "$CONFIG_YAML_ARG" ]; then
    SYNC_CONFIG_ARG="--sync-config $SYNC_CONFIG"
fi
SYNC_SKIP_OVERRIDES="${SYNC_SKIP_OVERRIDES:-0}"
SYNC_INCLUDE_CONFIG="${SYNC_INCLUDE_CONFIG:-0}"

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
export SYNC_METHOD SYNC_SEARCH_MIN SYNC_SEARCH_MAX SYNC_SEARCH_STEPS
export SIGNATURE_FORCE_WEIGHT SIGNATURE_STROKE_WEIGHT SIGNATURE_SMOOTH_WINDOW
export SHOW_TRUE_39MM
export CONFIG_YAML CONFIG_YAML_ARG
export SYNC_CONFIG SYNC_CONFIG_ARG SYNC_SKIP_OVERRIDES SYNC_INCLUDE_CONFIG

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

filtered_pairs_file="$pairs_file"
skip_file=""
include_file=""
processed_samples_file=""

write_sample_list() {
    local config_path="$1"
    local output_path="$2"
    if [[ "$config_path" == *.yaml || "$config_path" == *.yml ]]; then
        python - "$config_path" "$output_path" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
data = {}
if config_path.exists():
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
samples = data.get("samples", {}) or {}
with output_path.open("w") as f:
    for sample in samples.keys():
        if sample:
            f.write(f"{sample}\n")
PY
    else
        awk -F, 'NR > 1 && $1 != "" {gsub(/\r/, "", $1); print $1}' "$config_path" > "$output_path"
    fi
}

config_for_filter="$SYNC_CONFIG"
if [ -f "$CONFIG_YAML" ]; then
    config_for_filter="$CONFIG_YAML"
fi

if [ "$SYNC_INCLUDE_CONFIG" = "1" ] && [ -f "$config_for_filter" ]; then
    include_file=$(mktemp)
    write_sample_list "$config_for_filter" "$include_file"
    if [ -s "$include_file" ]; then
        filtered_pairs_file=$(mktemp)
        awk -F'|' 'NR==FNR {include[$1]=1; next} {
            video=$1;
            n=split(video, parts, "/");
            name=parts[n];
            sub(/\.mp4$/, "", name);
            if (include[name]) print $0
        }' "$include_file" "$pairs_file" > "$filtered_pairs_file"
    else
        echo "ERROR: SYNC_INCLUDE_CONFIG set but no samples found in $SYNC_CONFIG"
        exit 1
    fi
elif [ "$SYNC_SKIP_OVERRIDES" = "1" ] && [ -f "$config_for_filter" ]; then
    skip_file=$(mktemp)
    write_sample_list "$config_for_filter" "$skip_file"
    if [ -s "$skip_file" ]; then
        filtered_pairs_file=$(mktemp)
        awk -F'|' 'NR==FNR {skip[$1]=1; next} {
            video=$1;
            n=split(video, parts, "/");
            name=parts[n];
            sub(/\.mp4$/, "", name);
            if (!skip[name]) print $0
        }' "$skip_file" "$pairs_file" > "$filtered_pairs_file"
    fi
fi

processed_samples_file=$(mktemp)
awk -F'|' '{
    video=$1;
    n=split(video, parts, "/");
    name=parts[n];
    sub(/\.mp4$/, "", name);
    print name
}' "$filtered_pairs_file" > "$processed_samples_file"

total_videos=$(wc -l < "$filtered_pairs_file" | tr -d ' ')
echo "Found $total_videos videos to process"
echo ""
echo "Starting parallel processing with $NUM_JOBS jobs..."
echo "This will be much faster than sequential processing!"
echo ""

# Process in parallel
# Pass each line as a single argument to process_video
cat "$filtered_pairs_file" | parallel -j "$NUM_JOBS" --bar process_video {}

# Clean up temp file
rm "$pairs_file"
if [ "$filtered_pairs_file" != "$pairs_file" ]; then
    rm "$filtered_pairs_file"
fi
if [ -n "$skip_file" ]; then
    rm "$skip_file"
fi
if [ -n "$include_file" ]; then
    rm "$include_file"
fi

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
        if [ -n "$processed_samples_file" ]; then
            if ! grep -qx "$sample_name" "$processed_samples_file"; then
                continue
            fi
        fi
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

if [ -n "$processed_samples_file" ]; then
    rm "$processed_samples_file"
fi

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
