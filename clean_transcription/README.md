# Clean Transcription + Speaker Diarization

Deterministic transcript parsing and LLM-assisted speaker diarization correction for SHREC conversational transcripts.

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
├── script/
│   ├── parse.sh
│   ├── diarize.sh
│   └── summarize_diarazation_change.sh
```

---

# Pipeline

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
Human-readable diarization summary
```

---

# 1. Parse Transcript

## File

```text
parse_transcript.py
```

## Purpose

Parses raw transcript text into structured JSON.

Creates:
- word-level entries
- sentence-level chunks
- immutable timestamps
- original speaker labels

No LLM is used in this step.

---

## Run

### Single file

```bash
python parse_transcript.py example/sample.txt -o parsed --pretty
```

### Directory

```bash
python parse_transcript.py example -o parsed --pretty
```

---

## Output Example

```json
{
  "sentence_id": 0,
  "start_timestamp": "00:00:47",
  "end_timestamp": "00:00:48",
  "speaker_labels": ["User A"],
  "text": "Hello there."
}
```

---

# 2. Speaker Diarization Correction

## File

```text
speaker_diarization.py
```

## Purpose

Uses GPT to correct noisy speaker labels.

The model:
- does NOT rewrite transcript text
- does NOT modify timestamps
- only predicts:
  - `modified_speaker_label`
  - `modified_speaker_confidence`

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

Default:
- previous 10 sentences

---

## `pre_post_windowed_context`

Uses:
- previous corrected sentences
- current sentence
- future original sentences

Future labels may still be noisy.

Default:
- previous 10 sentences
- next 5 sentences

---

# Run Diarization

## Set API Key

```bash
export OPENAI_API_KEY="$(cat ~/.openai_api_key)"
```

---

## Default Method

```bash
python speaker_diarization.py \
  --input parsed \
  -o diarized_output
```

---

## Pre/Post Context Method

```bash
python speaker_diarization.py \
  --input parsed \
  -o diarized_output \
  --diarize-method pre_post_windowed_context
```

---

# Output Naming

Generated filenames include diarization method suffixes.

Example:

```text
conversation_001.pre_windowed_context.json
conversation_001.pre_post_windowed_context.json
```

---
# 3. Summarize Speaker Label Changes

## File

```text
print_diarization_change.py
```

## Purpose

Utility script for inspecting diarization updates and printing transcripts.

Features:

- Print sentences where:

  ```text
  original speaker label != modified speaker label
  ```

- Print surrounding context sentences
- Print using only current/final speaker labels
- Print full transcript grouped by speaker
- Print timestamped word-level transcripts using modified speaker labels

## Examples

```bash
python print_diarization_change.py sample.json
python print_diarization_change.py sample.json --current-only
python print_diarization_change.py sample.json --transcript
python print_diarization_change.py sample.json --timestamped-transcript
```

---

# Helper Scripts

## Parse

```bash
bash script/parse.sh
```

## Diarize

```bash
bash script/diarize.sh
```

## Summarize Changes

```bash
bash script/summarize_diarazation_change.sh
```

---

# Example End-to-End

```bash
# Step 1
python parse_transcript.py example -o parsed

# Step 2
python speaker_diarization.py \
  --input parsed \
  -o diarized_output

# Step 3
python print_diarization_change.py diarized_output
```

## Reprocess Multiple Annotation CSV Files

If you have multiple annotation CSV files to reprocess, you can use:

```bash
bash script/reprocess_csv.sh
```

In `script/reprocess_csv.sh`, set the list of annotation files here:

```bash
BASE_WORK_DIR="reprocess_runs"

ANNOTATION_FILES=(
  "annotations1.csv"
  "annotations2.csv"
  "annotations3.csv"
)
```

The script will:

1. iterate through each CSV in `ANNOTATION_FILES`
2. create a separate working directory under `reprocess_runs/`
3. run:
   - `parse_transcript.py`
   - `speaker_diarization.py`
   - `print_diarization_change.py --timestamped-transcript`
4. save a new CSV for each input file with a `new transcript` column

For example:

- `annotations1.csv` → `annotations1.reprocessed.csv`
- `annotations2.csv` → `annotations2.reprocessed.csv`

Each CSV gets its own intermediate folder, such as:

```text
reprocess_runs/annotations1/
reprocess_runs/annotations2/
```