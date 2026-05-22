export OPENAI_API_KEY="$(cat /mnt/ssd1/SHREC/openai_token)"

input_path="/mnt/ssd1/SHREC/analysis/parsed/example_transcript_dorm.json"
output_folder="output_example"

python speaker_diarization.py --input "$input_path" -o "$output_folder" --diarize-method "pre_post_windowed_context"