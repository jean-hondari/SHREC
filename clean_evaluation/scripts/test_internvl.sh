#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK_TYPE="${TASK_TYPE:-/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/output_datasets/nt_shrec_wellness_dorm_detection_agreed_cleaned_intervals_error_vs_competence.pickle}"
DATA_PATH="${DATA_PATH:-/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/output_datasets}"
IMAGES_DIR="${IMAGES_DIR:-/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm}"
CSV_PATH="${CSV_PATH:-/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm.reprocessed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./clean_evaluation/output}"
LOG_DIR="${LOG_DIR:-./clean_evaluation/logs}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/internvl_vlmeval_$(date +"%Y%m%d_%H%M%S").log"

echo "[STEP] Checking vlmeval + InternVL import..." | tee "$LOG_FILE"

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
try:
    from vlmeval.config import supported_VLM
    print("[OK] imported supported_VLM from vlmeval.config")
    print("[INFO] Available keys preview:")
    keys = list(supported_VLM.keys())
    preview = [k for k in keys if "InternVL" in k or "internvl" in k]
    print(preview if preview else keys[:20])
except Exception as e:
    print("[ERROR] Failed to import vlmeval or supported_VLM:", e)
    raise
PY

echo "[STEP] Running InternVL smoke test..." | tee -a "$LOG_FILE"

python3 ./clean_evaluation/run_eval.py \
  --task_type "$TASK_TYPE" \
  --model internvl \
  --data_path "$DATA_PATH" \
  --images_dir "$IMAGES_DIR" \
  --csv_path "$CSV_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --seed 0 \
  --temperature 0.0 \
  2>&1 | tee -a "$LOG_FILE"

echo "[DONE] InternVL vlmeval test complete. Log: $LOG_FILE" | tee -a "$LOG_FILE"