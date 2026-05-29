python3 clean_preprocess_step.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic.reprocessed.csv \
  --data_name nt_shrec_wellness_empathic \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic \
  --transcript_level exact \
  --buffer_seconds 5.0

python3 clean_preprocess_step.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm.reprocessed.csv \
  --data_name nt_shrec_wellness_dorm \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm \
  --transcript_level exact \
  --buffer_seconds 5.0

python3 clean_preprocess_step.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_home.reprocessed.csv \
  --data_name nt_shrec_wellness_home \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_home \
  --transcript_level exact \
  --buffer_seconds 5.0