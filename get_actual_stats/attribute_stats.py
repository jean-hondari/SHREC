import os
import sys
import ast
import argparse
from collections import defaultdict

import pandas as pd

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


ATTRIBUTE_ALIASES = {
    "Social Context &  Relationships": "Social Context & Relationships",
    "Social and Context Relationships": "Social Context & Relationships",
    "Social Context and Relationships": "Social Context & Relationships",
    "Social Relationships": "Social Context & Relationships",
    "Social Norms and Routines": "Social Norms & Routines",
    "Recognizing Social Norms including toxicity": "Social Norms & Routines",
}

def normalize_attribute_dict(attribute_dict):
    normalized = {name: False for name in ATTRIBUTE_NAMES}

    if not isinstance(attribute_dict, dict):
        return normalized

    for key, value in attribute_dict.items():
        mapped_key = ATTRIBUTE_ALIASES.get(key, key)
        if mapped_key in normalized:
            normalized[mapped_key] = bool(value)

    return normalized


def parse_annotation_cell(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    try:
        parsed = ast.literal_eval(cell)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def filter_valid_intervals(intervals):
    filtered = []
    for item in intervals:
        if not isinstance(item, dict):
            continue

        timestamp = item.get("timestamp", {})
        start = timestamp.get("start")
        end = timestamp.get("end")

        if start is None or end is None:
            continue

        item_copy = dict(item)
        item_copy["attribute"] = normalize_attribute_dict(item.get("attribute", {}))
        filtered.append(item_copy)

    return filtered


def get_annotation_lists_by_name(row):
    annotations_a = filter_valid_intervals(parse_annotation_cell(row.get("Annotations_A")))
    annotations_b = filter_valid_intervals(parse_annotation_cell(row.get("Annotations_B")))
    annotations_c = filter_valid_intervals(parse_annotation_cell(row.get("Annotations_C")))

    return {
        "Annotations_A": annotations_a,
        "Annotations_B": annotations_b,
        "Annotations_C": annotations_c,
    }


def get_first_two_non_empty_annotations(row):
    annotation_map = get_annotation_lists_by_name(row)
    non_empty = [value for value in annotation_map.values() if len(value) > 0]

    if len(non_empty) < 2:
        return None, None

    return non_empty[0], non_empty[1]


def get_first_two_non_empty_annotations_with_names(row):
    annotation_map = get_annotation_lists_by_name(row)
    non_empty = [(name, value) for name, value in annotation_map.items() if len(value) > 0]

    if len(non_empty) < 2:
        return None, None

    return non_empty[0], non_empty[1]


def get_overlap_groups_from_row(row):
    intervals1, intervals2 = get_first_two_non_empty_annotations(row)
    if intervals1 is None or intervals2 is None:
        return []

    return find_overlapping_interval_groups_pair(intervals1, intervals2)


def extract_agreed_attribute_samples_from_row(row):
    """
    Match repo logic for task_type == 'attribute':
    - use first two non-empty annotation lists
    - compute overlap groups
    - keep only groups of size 2
    - require agreement on error
    - merge attributes with OR
    """
    overlapping_groups = get_overlap_groups_from_row(row)
    agreed_samples = []

    for group in overlapping_groups:
        if len(group) != 2:
            continue

        ann1, ann2 = group[0], group[1]

        if ann1.get("error") == ann2.get("error"):
            agreed_dict = or_operation(
                normalize_attribute_dict(ann1.get("attribute", {})),
                normalize_attribute_dict(ann2.get("attribute", {})),
            )

            agreed_samples.append(
                {
                    "file_name": row.get("file_name", ""),
                    "timestamp": ann1.get("timestamp", {}),
                    "error": ann1.get("error"),
                    "attribute": agreed_dict,
                }
            )

    return agreed_samples


def extract_rationale_and_correction_samples_from_row(row):
    """
    Match repo logic for rationale/correction:
    - use overlap groups from first two non-empty annotators
    - take the first annotation in each overlap group
    - rationale counts if non-empty
    - correction counts if error == True and correction non-empty
    """
    overlapping_groups = get_overlap_groups_from_row(row)
    output = []

    for group in overlapping_groups:
        if len(group) == 0:
            continue

        sample = group[0]

        rationale = sample.get("rationale", "")
        correction = sample.get("correction", "")
        error_value = sample.get("error", None)

        rationale_non_empty = isinstance(rationale, str) and len(rationale.strip()) > 0
        correction_non_empty = isinstance(correction, str) and len(correction.strip()) > 0

        output.append(
            {
                "file_name": row.get("file_name", ""),
                "timestamp": sample.get("timestamp", {}),
                "error": error_value,
                "has_rationale": rationale_non_empty,
                "has_correction": bool(error_value is True and correction_non_empty),
            }
        )

    return output


def debug_disagreeing_error_rows(df, csv_path, verbose=False, max_examples=20):
    """
    Debug rows where the first two non-empty annotator lists contain overlapping
    segments with disagreeing error labels.

    Also check whether all three annotator columns are non-empty for those rows.
    """
    debug_summary = {
        "rows_with_all_three_nonempty": 0,
        "rows_with_error_disagreement": 0,
        "rows_with_error_disagreement_and_all_three_nonempty": 0,
        "total_disagreeing_overlap_groups": 0,
        "examples": [],
    }

    for row_idx, row in df.iterrows():
        annotation_map = get_annotation_lists_by_name(row)

        all_three_nonempty = all(
            len(annotation_map[col]) > 0
            for col in ["Annotations_A", "Annotations_B", "Annotations_C"]
        )

        if all_three_nonempty:
            debug_summary["rows_with_all_three_nonempty"] += 1

        first_pair = get_first_two_non_empty_annotations_with_names(row)
        if first_pair == (None, None):
            continue

        (name1, intervals1), (name2, intervals2) = first_pair
        overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

        disagreeing_groups = []
        for group in overlapping_groups:
            if len(group) != 2:
                continue

            ann1, ann2 = group[0], group[1]
            if ann1.get("error") != ann2.get("error"):
                disagreeing_groups.append(group)

        if len(disagreeing_groups) > 0:
            debug_summary["rows_with_error_disagreement"] += 1
            debug_summary["total_disagreeing_overlap_groups"] += len(disagreeing_groups)

            if all_three_nonempty:
                debug_summary["rows_with_error_disagreement_and_all_three_nonempty"] += 1

            if len(debug_summary["examples"]) < max_examples:
                debug_summary["examples"].append(
                    {
                        "csv_path": csv_path,
                        "row_index": row_idx,
                        "file_name": row.get("file_name", ""),
                        "first_two_nonempty": [name1, name2],
                        "all_three_nonempty": all_three_nonempty,
                        "num_disagreeing_groups": len(disagreeing_groups),
                        "annot_counts": {
                            "Annotations_A": len(annotation_map["Annotations_A"]),
                            "Annotations_B": len(annotation_map["Annotations_B"]),
                            "Annotations_C": len(annotation_map["Annotations_C"]),
                        },
                        "disagreeing_groups": [
                            {
                                "ann1_error": group[0].get("error"),
                                "ann2_error": group[1].get("error"),
                                "ann1_timestamp": group[0].get("timestamp"),
                                "ann2_timestamp": group[1].get("timestamp"),
                            }
                            for group in disagreeing_groups
                        ],
                    }
                )

    print(f"\n================ DEBUG: DISAGREEING ERROR ROWS ({csv_path}) ================\n")
    print(f"Rows with all three annotator columns non-empty: {debug_summary['rows_with_all_three_nonempty']}")
    print(f"Rows with disagreeing error overlap groups: {debug_summary['rows_with_error_disagreement']}")
    print(
        "Rows with disagreeing error overlap groups AND all three annotator columns non-empty: "
        f"{debug_summary['rows_with_error_disagreement_and_all_three_nonempty']}"
    )
    print(f"Total disagreeing overlap groups: {debug_summary['total_disagreeing_overlap_groups']}")

    if verbose and len(debug_summary["examples"]) > 0:
        print("\nExample rows:")
        for example in debug_summary["examples"]:
            print("-" * 80)
            print(f"row_index: {example['row_index']}")
            print(f"file_name: {example['file_name']}")
            print(f"first_two_nonempty: {example['first_two_nonempty']}")
            print(f"all_three_nonempty: {example['all_three_nonempty']}")
            print(f"annot_counts: {example['annot_counts']}")
            print(f"num_disagreeing_groups: {example['num_disagreeing_groups']}")
            for i, group in enumerate(example["disagreeing_groups"], 1):
                print(f"  disagreeing_group_{i}:")
                print(f"    ann1_error: {group['ann1_error']}")
                print(f"    ann2_error: {group['ann2_error']}")
                print(f"    ann1_timestamp: {group['ann1_timestamp']}")
                print(f"    ann2_timestamp: {group['ann2_timestamp']}")


def aggregate_csv(csv_path, aggregate_stats, per_file_summary):
    df = pd.read_csv(csv_path)

    file_summary = {
        "rows": len(df),
        "agreed_error_true_samples": 0,
        "agreed_error_false_samples": 0,
        "category_total_true_labels": defaultdict(int),
        "rationale_total": 0,
        "rationale_error_true": 0,
        "rationale_error_false": 0,
        "correction_total": 0,
        "correction_error_true": 0,
    }

    for _, row in df.iterrows():
        agreed_samples = extract_agreed_attribute_samples_from_row(row)

        for sample in agreed_samples:
            error_value = sample["error"]

            if error_value is True:
                aggregate_stats["samples_error_true"] += 1
                file_summary["agreed_error_true_samples"] += 1
            elif error_value is False:
                aggregate_stats["samples_error_false"] += 1
                file_summary["agreed_error_false_samples"] += 1
            else:
                continue

            for attr_name, is_true in sample["attribute"].items():
                if is_true:
                    if error_value is True:
                        aggregate_stats["error_true_per_attribute"][attr_name] += 1
                    elif error_value is False:
                        aggregate_stats["error_false_per_attribute"][attr_name] += 1

                    aggregate_stats["total_per_attribute"][attr_name] += 1
                    file_summary["category_total_true_labels"][attr_name] += 1

        rc_samples = extract_rationale_and_correction_samples_from_row(row)

        for sample in rc_samples:
            if sample["has_rationale"]:
                aggregate_stats["rationale_total"] += 1
                file_summary["rationale_total"] += 1

                if sample["error"] is True:
                    aggregate_stats["rationale_error_true"] += 1
                    file_summary["rationale_error_true"] += 1
                elif sample["error"] is False:
                    aggregate_stats["rationale_error_false"] += 1
                    file_summary["rationale_error_false"] += 1

            if sample["has_correction"]:
                aggregate_stats["correction_total"] += 1
                aggregate_stats["correction_error_true"] += 1
                file_summary["correction_total"] += 1
                file_summary["correction_error_true"] += 1

    per_file_summary[csv_path] = file_summary


def print_summary(aggregate_stats, per_file_summary):
    print("\n================ PER-FILE SUMMARY ================\n")
    for csv_path, summary in per_file_summary.items():
        print(f"CSV: {csv_path}")
        print(f"  Rows: {summary['rows']}")
        print(f"  Agreed samples with error=True: {summary['agreed_error_true_samples']}")
        print(f"  Agreed samples with error=False: {summary['agreed_error_false_samples']}")
        print("  Total true labels per category in this file:")
        for attr in ATTRIBUTE_NAMES:
            print(f"    - {attr}: {summary['category_total_true_labels'][attr]}")
        print("  Rationale stats:")
        print(f"    - total non-empty rationales: {summary['rationale_total']}")
        print(f"    - rationales on error=True samples: {summary['rationale_error_true']}")
        print(f"    - rationales on error=False samples: {summary['rationale_error_false']}")
        print("  Correction stats:")
        print(f"    - total non-empty corrections: {summary['correction_total']}")
        print(f"    - corrections on error=True samples: {summary['correction_error_true']}")
        print()

    print("\n================ AGGREGATED ATTRIBUTE SUMMARY ================\n")
    print(f"Total agreed samples with error=True: {aggregate_stats['samples_error_true']}")
    print(f"Total agreed samples with error=False: {aggregate_stats['samples_error_false']}")
    print()

    print("Per-attribute counts")
    print("-" * 100)
    print(f"{'Attribute':40s} {'error=True':>15s} {'error=False':>15s} {'total labels':>15s}")
    print("-" * 100)

    for attr in ATTRIBUTE_NAMES:
        true_count = aggregate_stats["error_true_per_attribute"][attr]
        false_count = aggregate_stats["error_false_per_attribute"][attr]
        total_count = aggregate_stats["total_per_attribute"][attr]
        print(f"{attr:40s} {true_count:15d} {false_count:15d} {total_count:15d}")

    print("-" * 100)

    print("\n================ AGGREGATED RATIONALE / CORRECTION SUMMARY ================\n")
    print("Rationale:")
    print(f"  total non-empty rationales: {aggregate_stats['rationale_total']}")
    print(f"  rationales on error=True samples: {aggregate_stats['rationale_error_true']}")
    print(f"  rationales on error=False samples: {aggregate_stats['rationale_error_false']}")
    print()
    print("Correction:")
    print(f"  total non-empty corrections: {aggregate_stats['correction_total']}")
    print(f"  corrections on error=True samples: {aggregate_stats['correction_error_true']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SHREC attribute/rationale/correction statistics across one or more CSV files."
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        required=True,
        help="One or more CSV paths",
    )
    parser.add_argument(
        "--debug_disagreeing_error",
        action="store_true",
        help="Debug rows where overlapping annotators disagree on error label.",
    )
    parser.add_argument(
        "--debug_verbose",
        action="store_true",
        help="Print verbose debug examples.",
    )
    args = parser.parse_args()

    aggregate_stats = {
        "samples_error_true": 0,
        "samples_error_false": 0,
        "error_true_per_attribute": defaultdict(int),
        "error_false_per_attribute": defaultdict(int),
        "total_per_attribute": defaultdict(int),
        "rationale_total": 0,
        "rationale_error_true": 0,
        "rationale_error_false": 0,
        "correction_total": 0,
        "correction_error_true": 0,
    }

    per_file_summary = {}

    for csv_path in args.data_paths:
        if args.debug_disagreeing_error:
            df = pd.read_csv(csv_path)
            debug_disagreeing_error_rows(
                df,
                csv_path=csv_path,
                verbose=args.debug_verbose,
            )

        aggregate_csv(csv_path, aggregate_stats, per_file_summary)

    print_summary(aggregate_stats, per_file_summary)


if __name__ == "__main__":
    main()