#!/usr/bin/env bash

#This is the file for preprocessing the transcripts for exact time alignment. It will run the main_vlm_get_data_exact.py script for each csv file and task type.
set -euo pipefail

# Edit these lists
csv_files=(
  "../shrec_hf/shrec_wellness_dorm.csv"
  "../shrec_hf/shrec_wellness_home.csv"
)

data_names=(
  "nt_shrec_wellness_dorm"
  "nt_shrec_wellness_home"
)

task_types=(
  "pre"
  "post"
  "detection"
  "attribute"
  "rationale"
  "correction"
  "attribute_agreed_multiple_subj"
  "detection_error_only"
)

# Optional shared args
transcript_level="exact"

# Sanity check: csv_files and data_names must align by index
if [ "${#csv_files[@]}" -ne "${#data_names[@]}" ]; then
  echo "Error: csv_files and data_names must have the same length."
  exit 1
fi

for i in "${!csv_files[@]}"; do
  csv="${csv_files[$i]}"
  name="${data_names[$i]}"

  for task in "${task_types[@]}"; do
    echo "Running: python main_vlm_get_data.py --data_path $csv --data_name $name --task_type $task --transcript_level $transcript_level"
    python3 main_vlm_get_data_exact.py \
      --data_path "$csv" \
      --data_name "$name" \
      --task_type "$task" \
      --transcript_level "$transcript_level"
  done
done