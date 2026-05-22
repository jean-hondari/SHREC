# Clean Transcription + Speaker Diarization

This folder contains the transcript cleaning pipeline for SHREC conversational transcripts.

Its purpose is to:

- parse timestamped transcript text into structured JSON
- preserve the original words, timestamps, and transcript text
- use an LLM only to correct speaker labels
- print the corrected transcript in human-readable form
- optionally reprocess annotation CSV files and save a new transcript column

The pipeline preserves:

- original words
- original timestamps
- original transcript text

The LLM is only used to predict corrected speaker labels.

---

# Repository Structure

```text
clean_transcription/
├── README.md
├── example/
├── output_example/
├── parsed/
├── parse_transcript.py
├── speaker_diarization.py
├── print_diarization_change.py
├── reprocess_transcript_csv.py
├── script/
│   ├── parse.sh
│   ├── diarize.sh
│   ├── summarize_diarazation_change.sh
│   └── reprocess_csv.sh
```

## Main Files

### `parse_transcript.py`
Parses raw transcript text into structured JSON with:

- word-level entries
- sentence-level chunks
- immutable timestamps
- original speaker labels

No LLM is used in this step.

### `speaker_diarization.py`
Runs speaker relabeling on parsed transcript JSON.

It:

- uses OpenAI to predict corrected speaker labels
- does not rewrite transcript text
- does not modify timestamps
- saves progress in real time
- supports retry and resume behavior

### `print_diarization_change.py`
Prints:

- changed speaker-label windows
- transcript grouped by speaker
- timestamped transcript using current speaker labels

### `reprocess_transcript_csv.py`
Runs the full pipeline for annotation CSV files that contain transcript rows, then writes a new CSV with a `new transcript` column.

---

# Full Pipeline

```text
Raw Transcript TXT
        ↓
parse_transcript.py
        ↓
Parsed JSON
        ↓
speaker_diarization.py
        ↓
Corrected speaker labels
        ↓
print_diarization_change.py
        ↓
Human-readable corrected transcript
```

---

# How to Run the Full Pipeline

## 1. Set API Key

```bash
export OPENAI_API_KEY="$(cat ~/.openai_api_key)"
```

## 2. Parse transcript text into JSON

### Single file

```bash
python parse_transcript.py example/sample.txt -o parsed --pretty
```

### Directory

```bash
python parse_transcript.py example -o parsed --pretty
```

## 3. Run speaker diarization

### Default method

```bash
python speaker_diarization.py \
  --input parsed \
  -o diarized_output
```

### Pre/post context method

```bash
python speaker_diarization.py \
  --input parsed \
  -o diarized_output \
  --diarize-method pre_post_windowed_context
```

## 4. Print the corrected transcript

### Show changed windows

```bash
python print_diarization_change.py diarized_output
```

### Show transcript grouped by speaker

```bash
python print_diarization_change.py diarized_output --transcript
```

### Show timestamped transcript

```bash
python print_diarization_change.py diarized_output --timestamped-transcript
```

---

# Run the CSV Reprocessing Pipeline

Use this when your annotation CSV contains one transcript per row, for example with columns like:

- `filename`
- `transcript`

The CSV pipeline will:

1. save each row's transcript as a text file named from `filename`
2. run parsing
3. run diarization
4. generate timestamped transcript output
5. write a new CSV with a `new transcript` column

## Run one CSV file directly

```bash
python reprocess_transcript_csv.py \
  --input-csv annotations.csv \
  --output-csv annotations.reprocessed.csv \
  --work-dir reprocess_runs/annotations \
  --model gpt-5.4 \
  --diarize-method pre_post_windowed_context
```

## Run multiple CSV files from the shell script

```bash
bash script/reprocess_csv.sh
```

In `script/reprocess_csv.sh`, define the list of CSV files:

```bash
BASE_WORK_DIR="reprocess_runs"

ANNOTATION_FILES=(
  "annotations1.csv"
  "annotations2.csv"
  "annotations3.csv"
)
```

For each CSV, the script creates a separate work folder and output CSV.

Example outputs:

- `annotations1.csv` → `annotations1.reprocessed.csv`
- `annotations2.csv` → `annotations2.reprocessed.csv`

Example intermediate folders:

```text
reprocess_runs/annotations1/
reprocess_runs/annotations2/
```

---

# Valid Speaker Labels

Only two labels are allowed:

```text
AI Agent
User A
```

Structured output schema enforcement prevents invalid labels.

---

# Diarization Methods

## `pre_windowed_context`

Uses:

- current sentence
- previous corrected sentences

Default context:

- previous 10 sentences

## `pre_post_windowed_context`

Uses:

- previous corrected sentences
- current sentence
- future original sentences

Future labels may still be noisy.

Default context:

- previous 10 sentences
- next 5 sentences

---

# Output Naming

Generated diarization filenames include the diarization method suffix.

Example:

```text
conversation_001.pre_windowed_context.json
conversation_001.pre_post_windowed_context.json
```

---

# Example: Run One Test End-to-End

If you want to test the pipeline on a small example first:

```bash
export OPENAI_API_KEY="$(cat ~/.openai_api_key)"

python parse_transcript.py example -o parsed --pretty

python speaker_diarization.py \
  --input parsed \
  -o output_example \
  --diarize-method pre_post_windowed_context \
  --model gpt-5.4

python print_diarization_change.py \
  output_example \
  --timestamped-transcript
```

This is the easiest way to verify that:

- parsing works
- diarization works
- transcript printing works

before running the CSV pipeline.

---

# Helper Scripts

## Parse example input

```bash
bash script/parse.sh
```

## Run diarization

```bash
bash script/diarize.sh
```

## Print timestamped transcript / summarize changes

```bash
bash script/summarize_diarazation_change.sh
```