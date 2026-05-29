#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import random

from utils_exact import or_operation, remove_redundant_strings_id_timestamp


SUPPORTED_TASKS = {
    "detection",
    "attribute",
    "attribute_agreed_multiple_subj",
    "rationale_error",
    "rationale_competence",
    "correction",
}


def load_clean_records(path):
    with open(path, "r") as f:
        return json.load(f)


def load_blocklist_video_ids(path):
    """
    Mandatory blocklist loader.

    Supported input:
    1. JSON list of strings:
        ["video1", "video2"]
    2. JSON list of dicts with video_id:
        [{"video_id": "video1"}, ...]
    """
    with open(path, "r") as f:
        data = json.load(f)

    blocked = set()

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                blocked.add(item)
            elif isinstance(item, dict) and "video_id" in item:
                blocked.add(item["video_id"])

    return blocked


def has_nonempty_text(value):
    return isinstance(value, str) and value.strip() != ""


def is_binary_error_label(value):
    return value is True or value is False


def count_true_attributes(attr_dict):
    if not isinstance(attr_dict, dict):
        return 0
    return sum(bool(v) for v in attr_dict.values())


def get_attr_pair(rec):
    attr_a = rec.get("annotator_a", {}).get("attribute")
    attr_b = rec.get("annotator_b", {}).get("attribute")

    if not isinstance(attr_a, dict) or not isinstance(attr_b, dict):
        return None, None

    return attr_a, attr_b


def print_count(stage_name, count):
    print(f"[INFO] {stage_name}: {count} samples left")


def filter_clean_records(clean_records, remove_no_transcript, blocked_video_ids):
    """
    Global filtering applied before task-specific construction.

    Included behavior:
    - ALWAYS remove samples whose video_id is in blocked_video_ids
    - optionally remove samples with empty transcript

    This filtering affects every task.
    """
    print_count("loaded clean records", len(clean_records))

    filtered = []
    removed_no_transcript = 0
    removed_blocked_video = 0

    for rec in clean_records:
        transcript = rec.get("transcript", "")
        video_id = rec.get("video_id")

        if video_id in blocked_video_ids:
            removed_blocked_video += 1
            continue

        if remove_no_transcript and not has_nonempty_text(transcript):
            removed_no_transcript += 1
            continue

        filtered.append(rec)

    print(f"[INFO] Removed blocked video_ids: {removed_blocked_video}")
    print_count("after blocked video_id filtering", len(clean_records) - removed_blocked_video)

    if remove_no_transcript:
        print(f"[INFO] Removed empty-transcript samples: {removed_no_transcript}")
    print_count("after optional no-transcript filtering", len(filtered))

    return filtered


def build_detection_dataset(clean_records):
    """
    Construct labels for binary error detection.

    Label construction:
    - target label = rec["error"]
    - keep only samples whose agreed detection label is exactly True or False
    - these are the segments where annotators already agreed on error status

    Included in output:
    - transcript
    - binary error label
    - annotator metadata for inspection
    """
    print("[INFO] Generating task: detection")
    processed = []
    skipped_non_binary = 0

    for rec in clean_records:
        error = rec.get("error")

        if not is_binary_error_label(error):
            skipped_non_binary += 1
            continue

        processed.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "transcript": rec.get("transcript", ""),
                "label": error,
                "error": error,
                "annotator_a_rationale": rec.get("annotator_a", {}).get("rationale", ""),
                "annotator_b_rationale": rec.get("annotator_b", {}).get("rationale", ""),
                "annotator_a_attribute": rec.get("annotator_a", {}).get("attribute"),
                "annotator_b_attribute": rec.get("annotator_b", {}).get("attribute"),
            }
        )

    print(f"[INFO] Removed non-binary error labels: {skipped_non_binary}")
    print_count("after detection label construction", len(processed))
    return processed


def build_attribute_dataset(clean_records):
    """
    Construct labels for attribute prediction.

    Task meaning:
    - the segment is already known to be either a social error or social competence
    - predict the relevant social attribute(s)

    Label construction:
    - require agreed binary detection label (True or False)
    - attribute label = OR across annotator attribute dicts

    Removed if:
    - error label is not binary
    - attribute dict is malformed
    """
    print("[INFO] Generating task: attribute")
    processed = []
    skipped_non_binary = 0
    skipped_bad_attr = 0

    for rec in clean_records:
        error = rec.get("error")
        if not is_binary_error_label(error):
            skipped_non_binary += 1
            continue

        attr_a, attr_b = get_attr_pair(rec)
        if attr_a is None or attr_b is None:
            skipped_bad_attr += 1
            continue

        merged_attr = or_operation(attr_a, attr_b)

        processed.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "transcript": rec.get("transcript", ""),
                "label": merged_attr,
                "error": error,
                "attribute": merged_attr,
                "annotator_a_attribute": attr_a,
                "annotator_b_attribute": attr_b,
                "annotator_a_rationale": rec.get("annotator_a", {}).get("rationale", ""),
                "annotator_b_rationale": rec.get("annotator_b", {}).get("rationale", ""),
            }
        )

    print(f"[INFO] Removed non-binary error labels: {skipped_non_binary}")
    print_count("after binary-error filtering", len(clean_records) - skipped_non_binary)
    print(f"[INFO] Removed malformed attribute labels: {skipped_bad_attr}")
    print_count("after attribute label construction", len(processed))
    print("[INFO] Attribute label is OR across annotators.")
    return processed


def build_attribute_agreed_multiple_subj_dataset(clean_records):
    """
    Construct labels for 'more than one social attribute involved' classification.

    Task meaning:
    - determine whether more than one social attribute is involved

    Label construction:
    - require agreed binary detection label
    - first construct merged attribute dict = OR across annotators
    - count how many attributes are True
    - target label:
        True  -> more than one attribute is True
        False -> exactly one attribute is True

    Removed if:
    - error label is not binary
    - attribute dict is malformed
    - zero attributes are True (because task is defined as multi-vs-single attribute involvement)
    """
    print("[INFO] Generating task: attribute_agreed_multiple_subj")
    processed = []
    skipped_non_binary = 0
    skipped_bad_attr = 0
    skipped_zero_true = 0

    for rec in clean_records:
        error = rec.get("error")
        if not is_binary_error_label(error):
            skipped_non_binary += 1
            continue

        attr_a, attr_b = get_attr_pair(rec)
        if attr_a is None or attr_b is None:
            skipped_bad_attr += 1
            continue

        merged_attr = or_operation(attr_a, attr_b)
        num_true = count_true_attributes(merged_attr)

        if num_true == 0:
            skipped_zero_true += 1
            continue

        label = num_true > 1

        processed.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "transcript": rec.get("transcript", ""),
                "label": label,
                "error": error,
                "num_true_attributes": num_true,
                "attribute": merged_attr,
                "annotator_a_attribute": attr_a,
                "annotator_b_attribute": attr_b,
                "annotator_a_rationale": rec.get("annotator_a", {}).get("rationale", ""),
                "annotator_b_rationale": rec.get("annotator_b", {}).get("rationale", ""),
            }
        )

    print(f"[INFO] Removed non-binary error labels: {skipped_non_binary}")
    print_count("after binary-error filtering", len(clean_records) - skipped_non_binary)
    print(f"[INFO] Removed malformed attribute labels: {skipped_bad_attr}")
    print_count("after valid-attribute filtering", len(clean_records) - skipped_non_binary - skipped_bad_attr)
    print(f"[INFO] Removed zero-attribute samples: {skipped_zero_true}")
    print_count("after attribute_agreed_multiple_subj label construction", len(processed))
    print("[INFO] Label=True means >1 true attribute; Label=False means exactly 1 true attribute.")
    print("[INFO] Attribute basis is OR across annotators.")
    return processed


def attributes_differ(attr1, attr2):
    """
    Used to avoid choosing distractors with identical attribute annotations.
    """
    return attr1 != attr2


def build_rationale_dataset(clean_records, target_error, task_name, num_distractors=5):
    """
    Construct labels and distractors for rationale selection.

    Task meaning:
    - explain why the segment is a social error or a social competence

    Pool restriction:
    - rationale_error      uses only social error samples (error == True)
    - rationale_competence uses only social competence samples (error == False)

    Label construction:
    - require agreed binary detection label equal to target_error
    - require non-empty rationale
    - target rationale = annotator A rationale
    - attribute label basis for filtering = OR across annotators

    Distractor construction:
    - sample only from the same pool (error-only or competence-only)
    - use other rationales as distractors
    - try to avoid identical attribute annotations
    - deduplicate exact repeated rationale strings
    - require at least num_distractors distractors
    """
    print(f"[INFO] Generating task: {task_name}")

    base_pool = []
    skipped_non_binary = 0
    skipped_wrong_error = 0
    skipped_empty_rationale = 0
    skipped_bad_attr = 0

    for rec in clean_records:
        error = rec.get("error")
        if not is_binary_error_label(error):
            skipped_non_binary += 1
            continue

        if error is not target_error:
            skipped_wrong_error += 1
            continue

        rationale = rec.get("annotator_a", {}).get("rationale", "")
        if not has_nonempty_text(rationale):
            skipped_empty_rationale += 1
            continue

        attr_a, attr_b = get_attr_pair(rec)
        if attr_a is None or attr_b is None:
            skipped_bad_attr += 1
            continue

        merged_attr = or_operation(attr_a, attr_b)

        base_pool.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "transcript": rec.get("transcript", ""),
                "label": rationale,
                "rationale": rationale,
                "error": error,
                "attribute": merged_attr,
                "annotator_a_rationale": rationale,
                "annotator_b_rationale": rec.get("annotator_b", {}).get("rationale", ""),
                "annotator_a_attribute": attr_a,
                "annotator_b_attribute": attr_b,
            }
        )

    print(f"[INFO] Removed non-binary error labels: {skipped_non_binary}")
    print_count("after binary-error filtering", len(clean_records) - skipped_non_binary)
    print(f"[INFO] Removed samples not in target pool", skipped_wrong_error)
    print_count("after target-pool filtering", len(clean_records) - skipped_non_binary - skipped_wrong_error)
    print(f"[INFO] Removed empty-rationale samples: {skipped_empty_rationale}")
    print_count(
        "after non-empty rationale filtering",
        len(clean_records) - skipped_non_binary - skipped_wrong_error - skipped_empty_rationale,
    )
    print(f"[INFO] Removed malformed attribute labels: {skipped_bad_attr}")
    print_count("base rationale pool", len(base_pool))

    processed = []
    removed_too_few_distractors = 0

    for sample in base_pool:
        candidate_pool = [
            other for other in base_pool
            if other["id"] != sample["id"] and attributes_differ(other["attribute"], sample["attribute"])
        ]

        distractor_texts = [other["rationale"] for other in candidate_pool]
        distractor_ids = [other["id"] for other in candidate_pool]
        distractor_timestamps = [other["timestamp"] for other in candidate_pool]

        distractor_texts, distractor_ids, distractor_timestamps = remove_redundant_strings_id_timestamp(
            distractor_texts, distractor_ids, distractor_timestamps
        )

        if len(distractor_texts) < num_distractors:
            removed_too_few_distractors += 1
            continue

        chosen_idx = list(range(len(distractor_texts)))
        random.shuffle(chosen_idx)
        chosen_idx = chosen_idx[:num_distractors]

        sample_out = dict(sample)
        sample_out["distractor_rationales"] = [distractor_texts[i] for i in chosen_idx]
        sample_out["distractor_ids"] = [distractor_ids[i] for i in chosen_idx]
        sample_out["distractor_timestamps"] = [distractor_timestamps[i] for i in chosen_idx]

        processed.append(sample_out)

    print(f"[INFO] Removed samples with too few distractors: {removed_too_few_distractors}")
    print_count(f"after {task_name} construction", len(processed))
    return processed


def build_correction_dataset(clean_records, num_distractors=5):
    """
    Construct labels and distractors for correction generation/selection.

    Task meaning:
    - correct a social error segment

    Pool restriction:
    - only social error samples (error == True)

    Label construction:
    - require non-empty correction text from annotator A
    - require non-empty rationale as requested
    - target correction = annotator A correction
    - attribute basis for filtering = OR across annotators

    Distractor construction:
    - sample correction distractors only from the social error pool
    - try to avoid identical attribute annotations
    - deduplicate exact repeated correction strings
    - require at least num_distractors distractors
    """
    print("[INFO] Generating task: correction")

    base_pool = []
    skipped_non_true_error = 0
    skipped_empty_correction = 0
    skipped_empty_rationale = 0
    skipped_bad_attr = 0

    for rec in clean_records:
        if rec.get("error") is not True:
            skipped_non_true_error += 1
            continue

        correction = rec.get("annotator_a", {}).get("correction", "")
        if not has_nonempty_text(correction):
            skipped_empty_correction += 1
            continue

        rationale = rec.get("annotator_a", {}).get("rationale", "")
        if not has_nonempty_text(rationale):
            skipped_empty_rationale += 1
            continue

        attr_a, attr_b = get_attr_pair(rec)
        if attr_a is None or attr_b is None:
            skipped_bad_attr += 1
            continue

        merged_attr = or_operation(attr_a, attr_b)

        base_pool.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "transcript": rec.get("transcript", ""),
                "label": correction,
                "correction": correction,
                "rationale": rationale,
                "error": True,
                "attribute": merged_attr,
                "annotator_a_correction": correction,
                "annotator_b_correction": rec.get("annotator_b", {}).get("correction", ""),
                "annotator_a_rationale": rationale,
                "annotator_b_rationale": rec.get("annotator_b", {}).get("rationale", ""),
            }
        )

    print(f"[INFO] Removed non-error samples: {skipped_non_true_error}")
    print_count("after social-error pool filtering", len(clean_records) - skipped_non_true_error)
    print(f"[INFO] Removed empty-correction samples: {skipped_empty_correction}")
    print_count(
        "after non-empty correction filtering",
        len(clean_records) - skipped_non_true_error - skipped_empty_correction,
    )
    print(f"[INFO] Removed empty-rationale samples: {skipped_empty_rationale}")
    print_count(
        "after non-empty rationale filtering",
        len(clean_records) - skipped_non_true_error - skipped_empty_correction - skipped_empty_rationale,
    )
    print(f"[INFO] Removed malformed attribute labels: {skipped_bad_attr}")
    print_count("base correction pool", len(base_pool))

    processed = []
    removed_too_few_distractors = 0

    for sample in base_pool:
        candidate_pool = [
            other for other in base_pool
            if other["id"] != sample["id"] and attributes_differ(other["attribute"], sample["attribute"])
        ]

        distractor_texts = [other["correction"] for other in candidate_pool]
        distractor_ids = [other["id"] for other in candidate_pool]
        distractor_timestamps = [other["timestamp"] for other in candidate_pool]

        distractor_texts, distractor_ids, distractor_timestamps = remove_redundant_strings_id_timestamp(
            distractor_texts, distractor_ids, distractor_timestamps
        )

        if len(distractor_texts) < num_distractors:
            removed_too_few_distractors += 1
            continue

        chosen_idx = list(range(len(distractor_texts)))
        random.shuffle(chosen_idx)
        chosen_idx = chosen_idx[:num_distractors]

        sample_out = dict(sample)
        sample_out["distractor_corrections"] = [distractor_texts[i] for i in chosen_idx]
        sample_out["distractor_ids"] = [distractor_ids[i] for i in chosen_idx]
        sample_out["distractor_timestamps"] = [distractor_timestamps[i] for i in chosen_idx]

        processed.append(sample_out)

    print(f"[INFO] Removed samples with too few distractors: {removed_too_few_distractors}")
    print_count("after correction construction", len(processed))
    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate task-specific pickle files from cleaned detection-agreed JSON.")
    parser.add_argument("--clean_json_path", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--output_dir", type=str, default="./clean_preprocess/output_datasets")
    parser.add_argument(
        "--remove_no_transcript",
        action="store_true",
        help="Remove samples whose transcript field is empty before task generation.",
    )
    parser.add_argument(
        "--exclude_video_ids_json",
        type=str,
        required=True,
        help="Mandatory JSON file containing video_ids to exclude.",
    )
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=5,
        help="Number of distractors for rationale/correction tasks.",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    clean_records = load_clean_records(args.clean_json_path)
    blocked_video_ids = load_blocklist_video_ids(args.exclude_video_ids_json)

    clean_records = filter_clean_records(
        clean_records,
        remove_no_transcript=args.remove_no_transcript,
        blocked_video_ids=blocked_video_ids,
    )

    if args.task_type == "detection":
        processed_dataset = build_detection_dataset(clean_records)

    elif args.task_type == "attribute":
        processed_dataset = build_attribute_dataset(clean_records)

    elif args.task_type == "attribute_agreed_multiple_subj":
        processed_dataset = build_attribute_agreed_multiple_subj_dataset(clean_records)

    elif args.task_type == "rationale_error":
        processed_dataset = build_rationale_dataset(
            clean_records,
            target_error=True,
            task_name="rationale_error",
            num_distractors=args.num_distractors,
        )

    elif args.task_type == "rationale_competence":
        processed_dataset = build_rationale_dataset(
            clean_records,
            target_error=False,
            task_name="rationale_competence",
            num_distractors=args.num_distractors,
        )

    elif args.task_type == "correction":
        processed_dataset = build_correction_dataset(
            clean_records,
            num_distractors=args.num_distractors,
        )

    else:
        raise ValueError(f"Unsupported task_type: {args.task_type}")

    clean_name = os.path.splitext(os.path.basename(args.clean_json_path))[0]
    pickle_path = os.path.join(args.output_dir, f"{clean_name}_{args.task_type}.pickle")

    with open(pickle_path, "wb") as handle:
        pickle.dump(processed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Saved task pickle: {pickle_path}")
    print(f"[INFO] Final sample count: {len(processed_dataset)}")


if __name__ == "__main__":
    main()