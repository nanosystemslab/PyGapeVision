#!/bin/bash
set -euo pipefail

LIST_FILE="${1:-scripts/batch/manual_review_list.txt}"

if [ ! -f "$LIST_FILE" ]; then
  echo "List file not found: $LIST_FILE"
  exit 1
fi

while IFS= read -r sample; do
  if [ -z "$sample" ]; then
    continue
  fi

  letter="${sample:0:1}"
  batch="Batch_${letter}1-${letter}25"
  video="data/Video/${batch}/${sample}.mp4"
  csv="data/Shimadzu/${batch}/${sample}.csv"
  out="results/batch/Batch_${letter}/${sample}"

  mkdir -p "$out"

  echo "Processing ${sample}..."
  poetry run python scripts/analyze_synced.py "$video" "$csv" \
    --frame-skip 10 \
    --fps 30 \
    --output-dir "$out" \
    --config configs/processing_config.yaml \
    --pixels-per-mm 12.103 \
    --sync-method multi_signature \
    --sync-search-min -30 \
    --sync-search-max 30 \
    --sync-search-steps 200 \
    --signature-force-weight 0.7 \
    --signature-stroke-weight 0.3 \
    --signature-smooth-window 5 \
    --show-true-39mm \
    --calculate-delta-gape \
    --use-datasheet-initial-gape \
    --interactive-init \
    --interactive-reacquire \
    --reacquire-miss-frames 15 \
    --reacquire-max-retries 5 \
    --search-radius 70 \
    --save-tracking-to-config
done < "$LIST_FILE"
