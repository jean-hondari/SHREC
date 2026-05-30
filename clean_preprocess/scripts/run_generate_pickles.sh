# Example usage:
# ./script/run_generate_pickles.sh


TASK_TYPES=(
  "error_vs_competence"
  "attribute"
  "attribute_agreed_multiple_subj"
  "rationale_error"
  "rationale_competence"
  "correction"
)

CLEAN_JSON_PATH="./clean_preprocess/cleaned_intervals/nt_shrec_wellness_empathic_detection_agreed_cleaned_intervals.json"
OUTPUT_DIR="./clean_preprocess/output_datasets"

for task in "${TASK_TYPES[@]}"; do
  echo "Generating pickle for task: ${task}"
  python3 generate_pickles_from_clean_json.py \
    --clean_json_path "${CLEAN_JSON_PATH}" \
    --task_type "${task}" \
    --output_dir "${OUTPUT_DIR}" \
    --remove_no_transcript \
    --exclude_video_ids_json /mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/cleaned_intervals/nt_shrec_wellness_empathic_detection_agreed_no_transcript_after_frame_cleaning.json
done

CLEAN_JSON_PATH="/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/cleaned_intervals/nt_shrec_wellness_dorm_detection_agreed_cleaned_intervals.json"


for task in "${TASK_TYPES[@]}"; do
  echo "Generating pickle for task: ${task}"
  python3 generate_pickles_from_clean_json.py \
    --clean_json_path "${CLEAN_JSON_PATH}" \
    --task_type "${task}" \
    --output_dir "${OUTPUT_DIR}" \
    --remove_no_transcript \
    --exclude_video_ids_json /mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/cleaned_intervals/nt_shrec_wellness_dorm_detection_agreed_no_transcript_after_frame_cleaning.json
done

CLEAN_JSON_PATH="/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/cleaned_intervals/nt_shrec_wellness_home_detection_agreed_cleaned_intervals.json"

for task in "${TASK_TYPES[@]}"; do
  echo "Generating pickle for task: ${task}"
  python3 generate_pickles_from_clean_json.py \
    --clean_json_path "${CLEAN_JSON_PATH}" \
    --task_type "${task}" \
    --output_dir "${OUTPUT_DIR}" \
    --remove_no_transcript \
    --exclude_video_ids_json /mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/cleaned_intervals/nt_shrec_wellness_home_detection_agreed_no_transcript_after_frame_cleaning.json
done