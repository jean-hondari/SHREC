python sanity_check/check_interval_transcript_and_frames.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm/data/shrec_wellness_dorm.csv \
  --mode singleton \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm \
  --transcript_level exact

python sanity_check/check_interval_transcript_and_frames.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_home/data/shrec_wellness_home.csv \
  --mode singleton \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_home \
  --transcript_level exact

python sanity_check/check_interval_transcript_and_frames.py \
  --data_path /mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic/data/shrec_wellness_empathic.csv \
  --mode singleton \
  --images_dir /mnt/ssd1/SHREC/shrec_data/shrec_wellness_empathic \
  --transcript_level exact