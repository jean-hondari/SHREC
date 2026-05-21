#!/usr/bin/env bash
set -euo pipefail

root="/mnt/ssd1/SHREC/shrec_benchmark"
# L_MODELS=(
#   "Llama-3.2-3B-Instruct"
# )

VL_MODELS=(
  "Llama-3.2-11B-Vision-Instruct"
)

TASK_TYPES_FOLDER="${root}/output_datasets_exact_seg"

CSV_FILES=(
  "../shrec_hf/shrec_wellness_dorm.csv"
  "../shrec_hf/shrec_wellness_home.csv"
)

IMAGE_PATHS=(
  "../shrec_hf/shrec_wellness_dorm"
  "../shrec_hf/shrec_wellness_home"
)

DATA_PATH="./output_datasets_exact_seg"
CONTEXT_WINDOW=15

find_matching_csv() {
  local pickle_name="$1"
  local pickle_base
  pickle_base="$(basename "$pickle_name" .pickle)"

  for csv_path in "${CSV_FILES[@]}"; do
    local csv_base
    csv_base="$(basename "$csv_path" .csv)"

    # Match by name containment in either direction, since they may not be exact
    if [[ "$pickle_base" == *"$csv_base"* ]] || [[ "$csv_base" == *"$pickle_base"* ]]; then
      echo "$csv_path"
      return 0
    fi
  done

  return 1
}

find_matching_image_dir() {
  local csv_path="$1"
  local csv_base
  csv_base="$(basename "$csv_path" .csv)"

  for image_dir in "${IMAGE_PATHS[@]}"; do
    local image_base
    image_base="$(basename "$image_dir")"

    if [[ "$image_base" == "$csv_base" ]]; then
      echo "$image_dir"
      return 0
    fi
  done

  return 1
}

run_model_on_all_pickles() {
  local model="$1"
  local is_vlm="$2"
  echo "Running model $model (is VLM: $is_vlm) on all pickles in $TASK_TYPES_FOLDER"

  find "$TASK_TYPES_FOLDER" -maxdepth 1 -type f -name "*.pickle" | sort | while read -r pickle_path; do
    pickle_file="$(basename "$pickle_path")"

    if ! csv_path="$(find_matching_csv "$pickle_file")"; then
      echo "Skipping $pickle_file: no matching CSV found"
      continue
    fi

    cmd=(
    env CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
    python main_vlm_exp.py
      --context_window "$CONTEXT_WINDOW"
      --model "$model"
      --data_path "$DATA_PATH"
      --task_type "$pickle_file"
      --csv_path "$csv_path"
    )

    if [[ "$is_vlm" == "true" ]]; then
      if ! images_dir="$(find_matching_image_dir "$csv_path")"; then
        echo "Skipping $pickle_file for VLM model $model: no matching image dir found for $csv_path"
        continue
      fi

      cmd+=(
        --video
        --images_dir "$images_dir"
      )
    fi

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
  done
}

for model in "${L_MODELS[@]}"; do
  run_model_on_all_pickles "$model" "false"
done

for model in "${VL_MODELS[@]}"; do
  run_model_on_all_pickles "$model" "true"
done