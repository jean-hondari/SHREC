#!/usr/bin/env bash
set -euo pipefail
TOKEN="$(tr -d '\n' < /mnt/ssd1/SHREC/hf_token)"
export HUGGING_FACE_HUB_TOKEN="$TOKEN"
export HF_TOKEN="$TOKEN"
huggingface-cli login --token "$TOKEN"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"


if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
  echo "[INFO] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  echo "[WARN] nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unchanged"
fi

DATA_PATH="${DATA_PATH:-/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/output_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ssd1/SHREC/SHREC/clean_evaluation/output}"
LOG_DIR="${LOG_DIR:-/mnt/ssd1/SHREC/SHREC/clean_evaluation/logs}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

DATASETS=("dorm" "home" "empathic")
TASK_NAMES=("error_vs_competence" "attribute" "rationale_error" "rationale_competence" "correction")

resolve_dataset_prefix() {
  local dataset="$1"
  case "$dataset" in
    dorm) echo "nt_shrec_wellness_dorm_detection_agreed_cleaned_intervals" ;;
    home) echo "nt_shrec_wellness_home_detection_agreed_cleaned_intervals" ;;
    empathic) echo "nt_shrec_wellness_empathic_detection_agreed_cleaned_intervals" ;;
    *) echo "[ERROR] Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  DATA_PREFIX="$(resolve_dataset_prefix "$dataset")"

  for task_name in "${TASK_NAMES[@]}"; do
    TASK_TYPE="${DATA_PREFIX}_${task_name}.pickle"
    LOG_FILE="${LOG_DIR}/llama_vlmeval_${dataset}_${task_name}_$(date +"%Y%m%d_%H%M%S").log"

    {
      echo "[INFO] model=llama-vlmeval"
      echo "[INFO] dataset=$dataset"
      echo "[INFO] task_name=$task_name"
      echo "[INFO] task_type=$TASK_TYPE"

      printf 'n\n' | python3 ./clean_evaluation/run_eval.py \
        --task_type "$TASK_TYPE" \
        --model llama \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --seed 0 \
        --temperature 0.0
    } 2>&1 | tee "$LOG_FILE"
  done
done