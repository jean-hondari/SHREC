#!/usr/bin/env python3
"""
Generate task-specific pickle files from the cleaned detection-agreed JSON file.

Extra filtering options:
- remove samples with no transcript
- remove samples whose video_id appears in a blocklist JSON/list

This is useful if no-transcript samples are known to have unreliable transcript alignment.
"""

import argparse
import json
import os
import pickle
import random

from utils_exact import (
    and_operation,
    or_operation,
    filter_dicts_without_subdict_tier2,
    remove_redundant_strings_id_timestamp,
)


def load_clean_records(path):
    with open(path, "r") as f:
        return json.load(f)


def load_blocklist_video_ids(path):
    """
    Load a blocklist file and return a set of video_ids.

    Supported input styles:
    1. JSON list of strings:
        ["file1", "file2"]
    2. JSON list of objects with video_id field:
        [{"video_id": "file1"}, ...]
    3. no-transcript JSON records from cleaning step:
        [{"video_id": "...", ...}, ...]
    """
    if path is None:
        return set()

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


def filter_clean_records(clean_records, remove_no_transcript=False, blocked_video_ids=None):
    blocked_video_ids = blocked_video_ids or set()

    filtered = []
    removed_no_transcript = 0
    removed_blocked_video = 0

    for rec in clean_records:
        transcript = rec.get("transcript", "")
        has_transcript = isinstance(transcript, str) and transcript.strip() != ""
        video_id = rec.get("video_id")

        if remove_no_transcript and not has_transcript:
            removed_no_transcript += 1
            continue

        if video_id in blocked_video_ids:
            removed_blocked_video += 1
            continue

        filtered.append(rec)

    print(f"[INFO] Records after optional filtering: {len(filtered)}")
    print(f"[INFO] Removed for empty transcript: {removed_no_transcript}")
    print(f"[INFO] Removed for blocked video_id: {removed_blocked_video}")

    return filtered


def check_all_false(dictionary):
    if not isinstance(dictionary, dict):
        return False
    return all(value is False for value in dictionary.values())


def build_detection_dataset(clean_records):
    processed_dataset = []
    for rec in clean_records:
        processed_dataset.append(
            {
                "video_id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "error": rec["error"],
                "transcription": rec["transcript"],
                "rationale": rec["annotator_a"].get("rationale", ""),
                "attribute": rec["annotator_a"].get("attribute"),
                "correction": rec["annotator_a"].get("correction", ""),
            }
        )
    return processed_dataset


def build_detection_error_only_dataset(clean_records):
    processed_dataset = []
    for rec in clean_records:
        if rec["error"] is not True:
            continue

        agreed_attribute = or_operation(
            rec["annotator_a"].get("attribute"),
            rec["annotator_b"].get("attribute"),
        )

        processed_dataset.append(
            {
                "video_id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "error": rec["error"],
                "transcription": rec["transcript"],
                "rationale": rec["annotator_a"].get("rationale", ""),
                "attribute": agreed_attribute,
                "correction": rec["annotator_a"].get("correction", ""),
            }
        )

    return processed_dataset


def resolve_attribute_agreement(rec, task_type):
    a = rec["annotator_a"]
    b = rec["annotator_b"]

    if a.get("error") != b.get("error"):
        return None

    attr_a = a.get("attribute")
    attr_b = b.get("attribute")

    if not isinstance(attr_a, dict) or not isinstance(attr_b, dict):
        return None

    if task_type == "attribute":
        return or_operation(attr_a, attr_b)

    if task_type == "attribute_agreed_multiple":
        agreed_dict = and_operation(attr_a, attr_b)
        if sum(list(agreed_dict.values())) >= 2:
            return agreed_dict
        return None

    if task_type == "attribute_agreed_multiple_subj":
        agreed_dict = and_operation(attr_a, attr_b)
        if sum(list(agreed_dict.values())) >= 2:
            return agreed_dict
        return None

    if task_type == "attribute_disagree":
        agreed_dict_or = or_operation(attr_a, attr_b)
        agreed_dict_and = and_operation(attr_a, attr_b)
        intersect_dict = and_operation(agreed_dict_or, agreed_dict_and)
        if sum(list(intersect_dict.values())) == 0:
            return agreed_dict_or
        return None

    return None


def build_attribute_dataset(clean_records, task_type):
    processed_dataset = []

    for rec in clean_records:
        agreed_attribute = resolve_attribute_agreement(rec, task_type)
        if agreed_attribute is None:
            continue

        processed_dataset.append(
            {
                "video_id": rec["video_id"],
                "id": rec["video_id"],
                "timestamp": rec["timestamp"],
                "error": rec["error"],
                "attribute": agreed_attribute,
                "transcription": rec["transcript"],
                "rationale": rec["annotator_a"].get("rationale", ""),
                "correction": rec["annotator_a"].get("correction", ""),
            }
        )

    return processed_dataset


def build_rationale_context_correction_dataset(clean_records, task_type):
    task_dataset_final = []

    for rec in clean_records:
        sample = {
            "video_id": rec["video_id"],
            "id": rec["video_id"],
            "timestamp": rec["timestamp"],
            "error": rec["error"],
            "attribute": rec["annotator_a"].get("attribute"),
            "transcript": rec["transcript"],
            "rationale": rec["annotator_a"].get("rationale", ""),
            "correction": rec["annotator_a"].get("correction", ""),
        }

        if task_type == "rationale" and len(sample["rationale"]) == 0:
            continue
        if task_type == "correction" and not (sample["error"] is True and len(sample["correction"]) > 0):
            continue
        if task_type == "context":
            pass

        task_dataset_final.append(sample)

    task_dataset_final_text = []
    for sample in task_dataset_final:
        attr = sample.get("attribute")
        if check_all_false(attr):
            continue

        random_subset = random.choices(task_dataset_final, k=len(task_dataset_final))
        others = filter_dicts_without_subdict_tier2(random_subset, attr)

        other_transcript_list = [other["transcript"] for other in others if other["error"] == sample["error"]]
        other_recovery_list = [other["correction"] for other in others if other["error"] == sample["error"]]
        other_reason_list = [other["rationale"] for other in others if other["error"] == sample["error"]]
        other_id_list = [other["id"] for other in others if other["error"] == sample["error"]]
        other_timestamp_list = [other["timestamp"] for other in others if other["error"] == sample["error"]]

        if task_type == "rationale":
            other_reason_list, other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(
                other_reason_list, other_id_list, other_timestamp_list
            )

        if task_type == "context":
            other_transcript_list, other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(
                other_transcript_list, other_id_list, other_timestamp_list
            )

        if task_type == "correction":
            other_recovery_list, other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(
                other_recovery_list, other_id_list, other_timestamp_list
            )

        if task_type == "rationale" and len(other_reason_list) < 5:
            continue
        if task_type == "correction" and len(other_recovery_list) < 5:
            continue
        if task_type == "context" and len(other_transcript_list) < 5:
            continue

        sample["other_reason_list"] = other_reason_list
        sample["other_recovery_list"] = other_recovery_list
        sample["other_transcript_list"] = other_transcript_list
        sample["other_id_list"] = other_id_list
        sample["other_timestamp_list"] = other_timestamp_list

        task_dataset_final_text.append(sample)

    return task_dataset_final_text


def split_speaker_blocks(transcript):
    if not isinstance(transcript, str) or transcript.strip() == "":
        return {}

    blocks = transcript.strip().split("\n\n")
    result = {}

    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        speaker = lines[0].strip()
        result[speaker] = "\n".join(lines)

    return result


def build_pre_post_dataset(clean_records, task_type):
    task_dataset_final = []

    for rec in clean_records:
        if rec["error"] is not False:
            continue

        speaker_blocks = split_speaker_blocks(rec["transcript"])

        if "User" not in speaker_blocks or "Agent" not in speaker_blocks:
            continue

        sample = {
            "video_id": rec["video_id"],
            "id": rec["video_id"],
            "timestamp": rec["timestamp"],
            "error": rec["error"],
            "attribute": rec["annotator_a"].get("attribute"),
            "transcript": rec["transcript"],
            "transcript_user": speaker_blocks["User"],
            "transcript_agent": speaker_blocks["Agent"],
            "rationale": rec["annotator_a"].get("rationale", ""),
            "correction": rec["annotator_a"].get("correction", ""),
        }
        task_dataset_final.append(sample)

    task_dataset_final_text = []
    for sample in task_dataset_final:
        attr = sample.get("attribute")
        if check_all_false(attr):
            continue

        others = random.choices(task_dataset_final, k=100)
        other_transcript_agent_list = [other["transcript_agent"] for other in others]
        other_transcript_user_list = [other["transcript_user"] for other in others]
        other_id_list = [other["id"] for other in others]
        other_timestamp_list = [other["timestamp"] for other in others]

        if task_type == "pre":
            other_transcript_user_list, other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(
                other_transcript_user_list, other_id_list, other_timestamp_list
            )

        if task_type == "post":
            other_transcript_agent_list, other_id_list, other_timestamp_list = remove_redundant_strings_id_timestamp(
                other_transcript_agent_list, other_id_list, other_timestamp_list
            )

        if task_type == "pre" and len(other_transcript_user_list) < 5:
            continue
        if task_type == "post" and len(other_transcript_agent_list) < 5:
            continue

        sample["other_transcript_agent_list"] = other_transcript_agent_list
        sample["other_transcript_user_list"] = other_transcript_user_list
        sample["other_id_list"] = other_id_list
        sample["other_timestamp_list"] = other_timestamp_list

        task_dataset_final_text.append(sample)

    return task_dataset_final_text


def main():
    parser = argparse.ArgumentParser(description="Generate task-specific pickle files from cleaned detection-agreed JSON.")
    parser.add_argument("--clean_json_path", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./clean_preprocess/output_datasets")
    parser.add_argument(
        "--remove_no_transcript",
        action="store_true",
        help="Remove samples whose transcript field is empty before task generation.",
    )
    parser.add_argument(
        "--exclude_video_ids_json",
        type=str,
        default=None,
        help="Optional JSON file containing video_ids to exclude. Useful for excluding no-transcript sample filenames.",
    )

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

    if args.task_type in ["debug", "detection"]:
        processed_dataset = build_detection_dataset(clean_records)

    elif args.task_type == "detection_error_only":
        processed_dataset = build_detection_error_only_dataset(clean_records)

    elif args.task_type in [
        "attribute",
        "attribute_disagree",
        "attribute_agreed_multiple",
        "attribute_agreed_multiple_subj",
    ]:
        processed_dataset = build_attribute_dataset(clean_records, args.task_type)

    elif args.task_type in ["rationale", "context", "correction"]:
        processed_dataset = build_rationale_context_correction_dataset(clean_records, args.task_type)

    elif args.task_type in ["pre", "post"]:
        processed_dataset = build_pre_post_dataset(clean_records, args.task_type)

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