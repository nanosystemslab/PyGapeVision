#!/bin/bash
# Quick progress checker for parallel batch processing

OUTPUT_BASE="results/batch"

echo "========================================"
echo "Batch Processing Progress"
echo "========================================"
echo ""

# Count completed files
COMPLETED=$(find "$OUTPUT_BASE" -name "*_synchronized.csv" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=198

# Calculate percentage
PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL)*100}")

echo "Files processed: $COMPLETED / $TOTAL ($PERCENT%)"
echo ""

# Show progress bar
COLS=50
FILLED=$(awk "BEGIN {printf \"%.0f\", ($COMPLETED/$TOTAL)*$COLS}")
printf "["
for i in $(seq 1 $FILLED); do printf "█"; done
for i in $(seq 1 $((COLS-FILLED))); do printf " "; done
printf "] $PERCENT%%\n"
echo ""

# Show batch breakdown
echo "Batches:"
for batch_dir in "$OUTPUT_BASE"/Batch_*; do
    if [ -d "$batch_dir" ]; then
        batch_name=$(basename "$batch_dir")
        count=$(find "$batch_dir" -name "*_synchronized.csv" 2>/dev/null | wc -l | tr -d ' ')
        printf "  %-10s: %2d files\n" "$batch_name" "$count"
    fi
done
echo ""

# Show most recent completions
echo "Most recent (last 5):"
find "$OUTPUT_BASE" -name "*_synchronized.csv" -type f 2>/dev/null | \
    xargs ls -t | head -5 | while read file; do
        sample=$(basename $(dirname "$file"))
        echo "  - $sample"
    done
echo ""
echo "========================================"
