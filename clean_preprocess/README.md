# clean_preprocess

This folder contains the cleaned preprocessing pipeline used before VLM evaluation.

The pipeline is separated into two stages:

1. **cleaning stage**
2. **task-specific pickle generation stage**

This separation is intentional so that:
- frame validation is done once
- transcript validation is done once
- downstream pickle generation can reuse the cleaned base intervals

---

## Files

### `clean_preprocess_step.py`

This is the first preprocessing step.

It:
- reads the annotation CSV
- finds overlapping intervals between annotators
- keeps only intervals where annotators agree on **detection**
- removes detection-agreed intervals that do not have any frames
- extracts transcript for each kept interval
- counts intervals that still have no transcript after frame cleaning
- saves cleaned JSON outputs for later use

Important behavior:
- transcript extraction may use `buffer_seconds`
- frame lookup uses only the raw annotation timestamp interval
- this script does **not** generate task pickle files

Typical outputs:
- cleaned detection-agreed interval JSON
- no-transcript-after-frame-cleaning JSON
- cleaning summary JSON

---

### `generate_pickles_from_clean_json.py`

This is the second preprocessing step.

It:
- loads the cleaned JSON produced by `clean_preprocess_step.py`
- builds task-specific datasets
- saves pickle files used by VLM evaluation

Important behavior:
- detection agreement is assumed to already be satisfied
- attribute agreement is computed here only when needed
- pre/post/rationale/context/correction reshaping is done here
- this script does **not** redo frame cleaning

Supported task types:
- `debug`
- `detection`
- `detection_error_only`
- `attribute`
- `attribute_disagree`
- `attribute_agreed_multiple`
- `attribute_agreed_multiple_subj`
- `rationale`
- `context`
- `correction`
- `pre`
- `post`

---

### `script/run_generate_pickles.sh`

Convenience shell script to generate pickle files for multiple task types from one cleaned JSON file.

Edit:
- `CLEAN_JSON_PATH`
- `OUTPUT_DIR`

before running if needed.

---

## Expected transcript format

The cleaned pipeline expects transcript text in the cleaned diarized timestamped format produced by the `clean_transcription` pipeline, for example:

```text
User A:
(00:00:12) hello (00:00:13) there.

AI Agent:
(00:00:15) hi (00:00:16) how (00:00:17) are (00:00:18) you?
```

During transcript extraction, the transcript is reformatted into sentence blocks like:

```text
User
00:12-00:14 hello there.

Agent
00:15-00:18 hi how are you?
```

---

## Workflow

### Step 1: Run cleaning

Example:

```bash
python3 clean_preprocess/clean_preprocess_step.py \
  --data_path ../shrec_data/shrec_wellness_empathic/data/train.csv \
  --data_name nt_shrec_wellness_empathic \
  --images_dir ../shrec_empathic \
  --transcript_level exact \
  --buffer_seconds 5.0
```

This produces the cleaned interval JSON.

---

### Step 2: Generate one pickle file

Example:

```bash
python3 clean_preprocess/generate_pickles_from_clean_json.py \
  --clean_json_path ./clean_preprocess/cleaned_intervals/nt_shrec_wellness_empathic_detection_agreed_cleaned_intervals.json \
  --task_type detection
```

---

### Step 3: Generate all pickle files

Example:

```bash
bash clean_preprocess/script/run_generate_pickles.sh
```

---

## Design summary

### Why detection agreement is done first
Detection agreement is the first annotator agreement stage.

So the cleaned base JSON is built only from:
- overlapping intervals
- where annotators agree on detection label
- and where frames actually exist

### Why attribute agreement is done later
Attribute agreement is only meaningful after detection agreement is already satisfied.

So attribute merging/filtering is done in `generate_pickles_from_clean_json.py`, not in `clean_preprocess_step.py`.

### Why no-transcript intervals are saved separately
Some intervals may:
- have valid frames
- but still produce empty transcript extraction

These are useful to inspect separately without contaminating the final cleaned base set silently.

### Why frame lookup does not use buffer
Transcript context and visual evidence serve different purposes.

So:
- transcript extraction may use `buffer_seconds`
- frame filtering always uses the true annotation interval only

---

## Notes

- If your CSV contains both original transcript and cleaned transcript, the pipeline should prefer the cleaned transcript column first.
- If the transcript parser returns empty results for all samples, check whether your CSV is using the cleaned diarized transcript column.
- The sentence timestamps are derived from the first and last word timestamps in each sentence.