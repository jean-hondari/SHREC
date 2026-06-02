#!/usr/bin/env bash
set -euo pipefail
export OPENAI_API_KEY="$(cat /mnt/ssd1/SHREC/openai_token)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

DATA_PATH="${DATA_PATH:-/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/output_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ssd1/SHREC/SHREC/clean_evaluation/output}"
LOG_DIR="${LOG_DIR:-/mnt/ssd1/SHREC/SHREC/clean_evaluation/logs}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

DATASETS=(
  "dorm"
  "home"
  "empathic"
)

TASK_NAMES=(
  "error_vs_competence"
  "attribute"
  "rationale_error"
  "rationale_competence"
  "correction"
)

resolve_dataset_paths() {
  local dataset="$1"

  case "$dataset" in
    dorm)
      DATA_PREFIX="nt_shrec_wellness_dorm_detection_agreed_cleaned_intervals"
      CSV_PATH="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm.reprocessed.csv"
      IMAGES_DIR="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm"
      ;;
    home)
      DATA_PREFIX="nt_shrec_wellness_home_detection_agreed_cleaned_intervals"
      CSV_PATH="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_home.reprocessed.csv"
      IMAGES_DIR="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_home"
      ;;
    empathic)
      DATA_PREFIX="nt_shrec_wellness_empathic_detection_agreed_cleaned_intervals"
      CSV_PATH="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic.reprocessed.csv"
      IMAGES_DIR="/mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic"
      ;;
    *)
      echo "[ERROR] Unknown dataset: $dataset"
      return 1
      ;;
  esac
}

echo "[STEP] Checking OpenAI package..."
python3 - <<'PY'
try:
    import openai
    print("[OK] openai import succeeded")
except Exception as e:
    print("[ERROR] openai import failed:", e)
    raise
PY

for dataset in "${DATASETS[@]}"; do
  resolve_dataset_paths "$dataset"

  for task_name in "${TASK_NAMES[@]}"; do
    TASK_TYPE="${DATA_PREFIX}_${task_name}.pickle"

    LOG_FILE="${LOG_DIR}/gpt4omini_lang_${dataset}_${task_name}_$(date +"%Y%m%d_%H%M%S").log"

    {
      echo "[INFO] model=gpt-4o-mini"
      echo "[INFO] dataset=$dataset"
      echo "[INFO] task_name=$task_name"
      echo "[INFO] task_type=$TASK_TYPE"
      echo "[INFO] csv_path=$CSV_PATH"
      echo "[INFO] images_dir=$IMAGES_DIR"
      echo "[INFO] data_path=$DATA_PATH"
      echo "[INFO] output_dir=$OUTPUT_DIR"
    } | tee "$LOG_FILE"

    if [[ ! -f "${DATA_PATH}/${TASK_TYPE}" ]]; then
      echo "[WARN] Missing pickle file, skipping: ${DATA_PATH}/${TASK_TYPE}" | tee -a "$LOG_FILE"
      continue
    fi

    if [[ ! -f "$CSV_PATH" ]]; then
      echo "[WARN] Missing CSV file, skipping: $CSV_PATH" | tee -a "$LOG_FILE"
      continue
    fi

    echo "[STEP] Running GPT-4o-mini language-only eval for dataset=$dataset task=$task_name" | tee -a "$LOG_FILE"

    python3 ./clean_evaluation/run_eval.py \
      --task_type "$TASK_TYPE" \
      --model gpt-4o-mini \
      --data_path "$DATA_PATH" \
      --images_dir "$IMAGES_DIR" \
      --csv_path "$CSV_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --seed 0 \
      --temperature 0.0 \
      2>&1 | tee -a "$LOG_FILE"

    echo "[DONE] Finished dataset=$dataset task=$task_name" | tee -a "$LOG_FILE"
  done
done

echo "[DONE] All GPT-4o-mini language-only runs finished."