TASK_TYPES=(
  "detection"
  "attribute"
  "attribute_agreed_multiple_subj"
  "rationale_error"
  "rationale_competence"
  "correction"
)

for task in "${TASK_TYPES[@]}"; do
  python3 get_pickle_stats.py \
    --input_dir ./clean_preprocess/output_datasets \
    --task_type "${task}"
done