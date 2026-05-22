#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="$(cat /mnt/ssd1/SHREC/openai_token)"

MODEL="gpt-5.5"
BASE_WORK_DIR="reprocess_runs"

ANNOTATION_FILES=(
  "/mnt/ssd1/SHREC/shrec_data/shrec_wellness_home/data/shrec_wellness_home.csv"
  "/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm/data/shrec_wellness_dorm.csv"
  "/mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic/data/shrec_wellness_empathic.csv"
)

mkdir -p "$BASE_WORK_DIR"

for INPUT_CSV in "${ANNOTATION_FILES[@]}"; do
  if [[ ! -f "$INPUT_CSV" ]]; then
    echo "SKIP missing file: $INPUT_CSV"
    continue
  fi

  BASENAME="$(basename "$INPUT_CSV")"
  STEM="${BASENAME%.*}"

  OUTPUT_CSV="${STEM}.reprocessed.csv"
  WORK_DIR="${BASE_WORK_DIR}/${STEM}"

  echo "============================================================"
  echo "Processing: $INPUT_CSV"
  echo "Output CSV: $OUTPUT_CSV"
  echo "Work dir:   $WORK_DIR"
  echo "============================================================"

  python reprocess_transcript_csv.py \
    --input-csv "$INPUT_CSV" \
    --output-csv "$OUTPUT_CSV" \
    --work-dir "$WORK_DIR" \
    --model "$MODEL" \
    --diarize-method pre_post_windowed_context \
    --history-size 10 \
    --post-size 5 \
    --max-retries 5

done