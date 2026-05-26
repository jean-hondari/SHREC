import argparse
import ast
from collections import defaultdict
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import find_overlapping_interval_groups_pair, or_operation


ATTRIBUTE_NAMES = [
    "Emotions",
    "Engagement",
    "Conversational Mechanics",
    "Knowledge State",
    "Intention",
    "Social Context & Relationships",
    "Social Norms & Routines",
]


def normalize_attribute_dict(attribute_dict):
    """
    Normalize slight naming inconsistencies across files.
    """
    if not isinstance(attribute_dict, dict):
        return {k: False for k in ATTRIBUTE_NAMES}

    normalized = {k: False for k in ATTRIBUTE_NAMES}

    alias_map = {
        "Social Context &  Relationships": "Social Context & Relationships",
        "Social and Context Relationships": "Social Context & Relationships",
        "Social Relationships": "Social Context & Relationships",
        "Social Norms and Routines": "Social Norms & Routines",
        "Recognizing Social Norms including toxicity": "Social Norms & Routines",
    }

    for key, value in attribute_dict.items():
        target_key = alias_map.get(key, key)
        if target_key in normalized:
            normalized[target_key] = bool(value)

    return normalized


def load_annotation_list(cell_value):
    if pd.isna(cell_value):
        return []
    if isinstance(cell_value, list):
        return cell_value
    try:
        parsed = ast.literal_eval(cell_value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def valid_intervals(intervals):
    cleaned = []
    for item in intervals:
        ts = item.get("timestamp", {})
        if ts.get("start") is not None and ts.get("end") is not None:
            item = dict(item)
            item["attribute"] = normalize_attribute_dict(item.get("attribute", {}))
            cleaned.append(item)
    return cleaned


def build_agreed_overlap_groups(row):
    """
    Mirrors the repo logic:
    - use first two non-empty annotator lists
    - group by overlap with utils.find_overlapping_interval_groups_pair
    - keep only groups of length 2
    - require annotator agreement on error label
    - merge attribute labels with OR
    """
    anots = []

    annotations_a = load_annotation_list(row["Annotations_A"])
    annotations_b = load_annotation_list(row["Annotations_B"])
    annotations_c = load_annotation_list(row["Annotations_C"])

    if len(annotations_a) > 0:
        anots.append(valid_intervals(annotations_a))
    if len(annotations_b) > 0:
        anots.append(valid_intervals(annotations_b))
    if len(annotations_c) > 0:
        anots.append(valid_intervals(annotations_c))

    if len(anots) < 2:
        return []

    intervals1 = anots[0]
    intervals2 = anots[1]

    overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

    agreed_groups = []
    for group in overlapping_groups:
        if len(group) != 2:
            continue

        ann1, ann2 = group[0], group[1]

        # Same rule as main_vlm_get_data.py for attribute task
        if ann1["error"] == ann2["error"]:
            agreed_dict = or_operation(
                normalize_attribute_dict(ann1.get("attribute", {})),
                normalize_attribute_dict(ann2.get("attribute", {})),
            )

            agreed_groups.append(
                {
                    "file_name": row["file_name"],
                    "timestamp": ann1["timestamp"],
                    "error": bool(ann1["error"]),
                    "attribute": agreed_dict,
                }
            )

    return agreed_groups


def compute_statistics(df):
    overall = {
        "social_error": defaultdict(int),
        "social_competency": defaultdict(int),
    }

    sample_counts = {
        "social_error": 0,
        "social_competency": 0,
    }

    all_samples = []

    for _, row in df.iterrows():
        agreed_samples = build_agreed_overlap_groups(row)

        for sample in agreed_samples:
            label_name = "social_error" if sample["error"] else "social_competency"
            sample_counts[label_name] += 1

            for attr_name, attr_value in sample["attribute"].items():
                if attr_value is True:
                    overall[label_name][attr_name] += 1

            all_samples.append(sample)

    return overall, sample_counts, all_samples


def print_statistics(overall, sample_counts):
    print("\n=== Overall Statistics Per Attribute ===\n")

    print(f"Total agreed social error samples: {sample_counts['social_error']}")
    print(f"Total agreed social competency samples: {sample_counts['social_competency']}\n")

    print("Per-attribute counts where attribute == True")
    print("-" * 80)
    print(f"{'Attribute':40s} {'Error=True':>15s} {'Competency=True':>20s}")
    print("-" * 80)

    for attr in ATTRIBUTE_NAMES:
        error_count = overall["social_error"][attr]
        competency_count = overall["social_competency"][attr]
        print(f"{attr:40s} {error_count:15d} {competency_count:20d}")

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to SHREC CSV file")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-sample agreed attribute labels",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    overall, sample_counts, all_samples = compute_statistics(df)
    print_statistics(overall, sample_counts)

    if args.output_csv is not None:
        rows = []
        for sample in all_samples:
            row = {
                "file_name": sample["file_name"],
                "start": sample["timestamp"]["start"],
                "end": sample["timestamp"]["end"],
                "error": sample["error"],
            }
            row.update(sample["attribute"])
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-sample statistics to: {args.output_csv}")


if __name__ == "__main__":
    main()