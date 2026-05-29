#!/usr/bin/env python3
import argparse
import os
import pickle
from collections import defaultdict


KNOWN_TASK_TYPES = [
  "detection",
  "attribute",
  "attribute_agreed_multiple_subj",
  "rationale_error",
  "rationale_competence",
  "correction"
]


def infer_task_type_from_filename(filename):
    """
    Infer task type from filename suffix before .pickle.

    We sort by descending length so that e.g.
    'attribute_agreed_multiple_subj' matches before 'attribute'.
    """
    stem = filename[:-7] if filename.endswith(".pickle") else filename

    for task_type in sorted(KNOWN_TASK_TYPES, key=len, reverse=True):
        if stem.endswith(f"_{task_type}"):
            return task_type

    return None


def load_pickle_count(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return len(data)
    return 1


def main():
    parser = argparse.ArgumentParser(description="Get total statistics for pickle files in output_datasets.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./clean_preprocess/output_datasets",
        help="Directory containing pickle files.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Optional: only count pickle files whose filename suffix matches this task type.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    counts_by_task = defaultdict(list)

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".pickle"):
            continue

        task_type = infer_task_type_from_filename(filename)
        if task_type is None:
            continue

        if args.task_type is not None and task_type != args.task_type:
            continue

        full_path = os.path.join(args.input_dir, filename)
        count = load_pickle_count(full_path)
        counts_by_task[task_type].append((filename, count))

    if not counts_by_task:
        print("[INFO] No matching pickle files found.")
        return

    grand_total = 0

    for task_type in sorted(counts_by_task.keys()):
        print("\n" + "=" * 80)
        print(f"Task type: {task_type}")
        print("=" * 80)

        task_total = 0
        for filename, count in counts_by_task[task_type]:
            print(f"{filename}: {count}")
            task_total += count

        print(f"TOTAL ({task_type}): {task_total}")
        grand_total += task_total

    print("\n" + "=" * 80)
    print(f"GRAND TOTAL: {grand_total}")
    print("=" * 80)


if __name__ == "__main__":
    main()