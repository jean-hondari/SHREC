#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


TIMESTAMP_WORD_RE = re.compile(r"\((\d{2}:\d{2}:\d{2})\)\s*([^\s()]+)")
SPEAKER_RE = re.compile(r"^\s*([^:\n]+):\s*$")
END_PUNCT = {".", "?", "!"}


def parse_labeled_transcript(text: str) -> Dict[str, Any]:
    words: List[Dict[str, Any]] = []
    current_speaker: Optional[str] = None
    word_id = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        speaker_match = SPEAKER_RE.match(stripped)
        if speaker_match and "(" not in stripped:
            current_speaker = speaker_match.group(1).strip()
            continue

        if current_speaker is None:
            continue

        for ts, word in TIMESTAMP_WORD_RE.findall(stripped):
            words.append(
                {
                    "word_id": word_id,
                    "word": word,
                    "timestamp": ts,
                    "speaker_label": current_speaker,
                    "line_no": line_no,
                }
            )
            word_id += 1

    return {
        "metadata": {
            "num_words": len(words),
        },
        "words": words,
        "sentences": build_sentences(words),
    }


def build_sentences(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sentences: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    sentence_id = 0

    for i, word_obj in enumerate(words):
        current.append(word_obj)

        word = word_obj["word"]
        ends_sentence = word.endswith(tuple(END_PUNCT))

        next_word = words[i + 1] if i + 1 < len(words) else None
        speaker_changes_next = (
            next_word is not None
            and next_word["speaker_label"] != word_obj["speaker_label"]
        )

        if ends_sentence or speaker_changes_next:
            sentences.append(make_sentence(sentence_id, current))
            sentence_id += 1
            current = []

    if current:
        sentences.append(make_sentence(sentence_id, current))

    return sentences


def make_sentence(sentence_id: int, sentence_words: List[Dict[str, Any]]) -> Dict[str, Any]:
    speaker_labels = list(dict.fromkeys(w["speaker_label"] for w in sentence_words))

    return {
        "sentence_id": sentence_id,
        "start_timestamp": sentence_words[0]["timestamp"],
        "end_timestamp": sentence_words[-1]["timestamp"],
        "speaker_labels": speaker_labels,
        "word_ids": [w["word_id"] for w in sentence_words],
        "text": " ".join(w["word"] for w in sentence_words),
        "words": sentence_words,
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".parsed.json")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse labeled word-timestamp transcript into immutable words and deterministic sentence chunks."
    )

    parser.add_argument(
        "input",
        help="Input transcript file or directory.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory where parsed JSON files will be saved.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parsed files.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # collect files
    if input_path.is_file():
        input_files = [input_path]

    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*"))

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    total_files = 0

    for file_path in input_files:
        if not file_path.is_file():
            continue

        output_path = output_dir / f"{file_path.stem}.json"

        if output_path.exists() and not args.overwrite:
            print(f"SKIP already exists: {output_path}")
            continue

        try:
            text = file_path.read_text(encoding="utf-8")

            parsed = parse_labeled_transcript(text)

            output_path.write_text(
                json.dumps(
                    parsed,
                    ensure_ascii=False,
                    indent=2 if args.pretty else None,
                ),
                encoding="utf-8",
            )

            print(f"SAVED: {output_path}")
            print(f"  Words: {len(parsed['words'])}")
            print(f"  Sentences: {len(parsed['sentences'])}")

            total_files += 1

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")

    print("\nDONE")
    print(f"Processed files: {total_files}")
    
if __name__ == "__main__":
    main()
