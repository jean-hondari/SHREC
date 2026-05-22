#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_original_label(sentence: Dict[str, Any]) -> str:
    labels = sentence.get("speaker_labels", [])

    if isinstance(labels, list):
        if len(labels) == 1:
            return str(labels[0]).strip()
        return ",".join(str(x).strip() for x in labels)

    return str(labels).strip()


def get_modified_label(sentence: Dict[str, Any]) -> str:
    return str(sentence.get("modified_speaker_label", "")).strip()


def get_changed_sentence_ids(sentences: List[Dict[str, Any]]) -> List[int]:
    changed_ids = []

    for i, sentence in enumerate(sentences):
        original = get_original_label(sentence)
        modified = get_modified_label(sentence)

        if modified and original != modified:
            print("note")
            changed_ids.append(i)

    return changed_ids


def format_sentence(sentence: Dict[str, Any], changed: bool) -> str:
    sid = sentence.get("sentence_id", "N/A")
    original = get_original_label(sentence)
    modified = get_modified_label(sentence)
    text = sentence.get("text", "")

    marker = " (*changed)" if changed else ""

    return f"[{sid}] {original} -> {modified}{marker}: {text}"


def print_changed_windows(sentences: List[Dict[str, Any]], changed_indices: List[int], context: int) -> None:
    changed_set = set(changed_indices)

    for idx in changed_indices:
        start = max(0, idx - context)
        end = min(len(sentences), idx + context + 1)

        print("=" * 100)
        print(f"Changed sentence_id={sentences[idx].get('sentence_id', 'N/A')}")
        print("-" * 100)

        for j in range(start, end):
            print(format_sentence(sentences[j], j in changed_set))

        print()


def collect_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(path.glob("*.json"))

    raise FileNotFoundError(f"Path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print sentences where speaker_labels and modified_speaker_label do not match."
    )
    parser.add_argument(
        "input",
        help="Input JSON file or directory.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Number of sentences before and after each changed sentence.",
    )

    args = parser.parse_args()

    files = collect_json_files(Path(args.input))

    total_changed = 0
    total_sentences = 0

    for file_path in files:
        data = load_json(file_path)
        sentences = data.get("sentences", [])

        changed_indices = get_changed_sentence_ids(sentences)

        print("\n" + "#" * 100)
        print(f"FILE: {file_path}")
        print("#" * 100)

        print_changed_windows(sentences, changed_indices, args.context)

        print(f"File summary: {len(changed_indices)} / {len(sentences)} sentences changed")

        total_changed += len(changed_indices)
        total_sentences += len(sentences)

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)
    print(f"Total files: {len(files)}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total changed sentences: {total_changed}")


if __name__ == "__main__":
    main()