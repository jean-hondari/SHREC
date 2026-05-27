#!/usr/bin/env bash

#This is the file for preprocessing the transcripts for exact time alignment. It will run the main_vlm_get_data_exact.py script for each csv file and task type.
set -euo pipefail

# Edit these lists
csv_files=(
  "../shrec_data/shrec_wellness_dorm/data/train.csv"
  "../shrec_data/shrec_wellness_home/data/train.csv"
)

data_names=(
  "shrec_wellness_dorm"
  "shrec_wellness_home"
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


# Sanity check: csv_files and data_names must align by index
if [ "${#csv_files[@]}" -ne "${#data_names[@]}" ]; then
  echo "Error: csv_files and data_names must have the same length."
  exit 1
fi

for i in "${!csv_files[@]}"; do
  csv="${csv_files[$i]}"
  name="${data_names[$i]}"

  for task in "${task_types[@]}"; do
    echo "Running: python analyze_transcript_duration_delta.py --data_path $csv --data_name $name --task_type $task"
    python ./sanity_check/analyze_transcript_duration_delta.py \
      --data_path "$csv" \
      --data_name "$name" \
      --task_type "$task" \
      --plot_path ./analysis/duration_delta_plots \
      --csv_path ./analysis/duration_delta_csv
  done
done