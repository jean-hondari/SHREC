#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configurable paths
# =========================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK_TYPE="${TASK_TYPE:-nt_shrec_wellness_dorm_detection_agreed_cleaned_intervals_detection.pickle}"
MODEL="${MODEL:-internvl}"
DATA_PATH="${DATA_PATH:-/mnt/ssd1/SHREC/SHREC/clean_preprocess/clean_preprocess/output_datasets}"
IMAGES_DIR="${IMAGES_DIR:-/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm}"
CSV_PATH="${CSV_PATH:-/mnt/ssd1/SHREC/shrec_data/shrec_wellness_dorm.reprocessed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./clean_evaluation/output}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LOG_DIR="${LOG_DIR:-./clean_evaluation/logs}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/sanitycheck_${MODEL}_$(basename "$TASK_TYPE" .pickle)_${TIMESTAMP}.log"

echo "[INFO] Repo root: $REPO_ROOT" | tee "$LOG_FILE"
echo "[INFO] Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# =========================
# Basic file checks
# =========================
echo "[STEP] Checking required files..." | tee -a "$LOG_FILE"

test -f "./clean_evaluation/run_eval.py"
test -f "./clean_evaluation/frame_sampling.py"
test -f "./clean_evaluation/model_adapters.py"
test -f "./clean_evaluation/exp_llm.py"
test -d "$DATA_PATH"
test -f "${DATA_PATH}/${TASK_TYPE}"

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "[ERROR] IMAGES_DIR does not exist: $IMAGES_DIR" | tee -a "$LOG_FILE"
  exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERROR] CSV_PATH does not exist: $CSV_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[OK] Required files/paths exist." | tee -a "$LOG_FILE"

# =========================
# Python import verification
# =========================
echo "[STEP] Verifying Python imports..." | tee -a "$LOG_FILE"

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import importlib

mods = [
    "json",
    "pickle",
    "argparse",
    "pandas",
    "tqdm",
]

for m in mods:
    importlib.import_module(m)

print("[OK] Base Python imports succeeded.")
PY

# Optional dependency checks
echo "[STEP] Checking optional model dependencies..." | tee -a "$LOG_FILE"

python3 - <<'PY' 2>&1 | tee -a "$LOG_FILE"
optional_mods = [
    "openai",
    "google.generativeai",
]

for m in optional_mods:
    try:
        __import__(m)
        print(f"[OK] Optional dependency available: {m}")
    except Exception as e:
        print(f"[WARN] Optional dependency missing or failed import: {m} :: {e}")

try:
    import transformers
    print("[OK] Optional dependency available: transformers")
except Exception as e:
    print(f"[WARN] Optional dependency missing or failed import: transformers :: {e}")

try:
    import vlmeval
    print("[OK] Optional dependency available: vlmeval")
except Exception as e:
    print(f"[WARN] Optional dependency missing or failed import: vlmeval :: {e}")
PY

# =========================
# Print run config
# =========================
echo "[STEP] Smoke test configuration:" | tee -a "$LOG_FILE"
echo "  TASK_TYPE=$TASK_TYPE" | tee -a "$LOG_FILE"
echo "  MODEL=$MODEL" | tee -a "$LOG_FILE"
echo "  DATA_PATH=$DATA_PATH" | tee -a "$LOG_FILE"
echo "  IMAGES_DIR=$IMAGES_DIR" | tee -a "$LOG_FILE"
echo "  CSV_PATH=$CSV_PATH" | tee -a "$LOG_FILE"
echo "  OUTPUT_DIR=$OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "  SEED=$SEED" | tee -a "$LOG_FILE"
echo "  TEMPERATURE=$TEMPERATURE" | tee -a "$LOG_FILE"

# =========================
# Run evaluation
# =========================
echo "[STEP] Running smoke test evaluation..." | tee -a "$LOG_FILE"
echo "[INFO] You will be prompted to choose frame sampling interval in seconds." | tee -a "$LOG_FILE"

python3 ./clean_evaluation/run_eval.py \
  --task_type "$TASK_TYPE" \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --images_dir "$IMAGES_DIR" \
  --csv_path "$CSV_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --temperature "$TEMPERATURE" \
  2>&1 | tee -a "$LOG_FILE"

echo "[DONE] Smoke test finished." | tee -a "$LOG_FILE"
echo "[DONE] Check log: $LOG_FILE" | tee -a "$LOG_FILE"