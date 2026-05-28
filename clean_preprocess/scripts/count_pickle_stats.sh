TASK_TYPES=(
  "debug"
  "detection"
  "detection_error_only"
  "attribute"
  "attribute_disagree"
  "attribute_agreed_multiple"
  "attribute_agreed_multiple_subj"
  "rationale"
  "context"
  "correction"
  "pre"
  "post"
)

for task in "${TASK_TYPES[@]}"; do
  python3 get_pickle_stats.py \
    --input_dir ./clean_preprocess/output_datasets \
    --task_type "${task}"
done