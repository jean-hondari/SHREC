#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os

import pandas as pd

from utils_exact import (
    find_overlapping_interval_groups_pair,
    get_frame_paths,
    get_transcript,
)


def safe_literal_eval(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        return ast.literal_eval(value)
    return value


def is_non_empty_text(text):
    return isinstance(text, str) and text.strip() != ""


def get_video_id(row):
    for key in ["file_name", "video_id", "id"]:
        if key in row and pd.notna(row[key]):
            return row[key]
    return "<unknown>"


def get_full_transcript(row):
    """
    Prefer cleaned transcript if available.
    """
    for key in ["new transcript", "new_transcript", "transcript", "transcription"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    return ""


def normalize_annotations(row):
    annotations_a = safe_literal_eval(row.get("Annotations_A", []))
    annotations_b = safe_literal_eval(row.get("Annotations_B", []))
    annotations_c = safe_literal_eval(row.get("Annotations_C", []))

    annot_lists = []
    if isinstance(annotations_a, list) and annotations_a:
        annot_lists.append(annotations_a)
    if isinstance(annotations_b, list) and annotations_b:
        annot_lists.append(annotations_b)
    if isinstance(annotations_c, list) and annotations_c:
        annot_lists.append(annotations_c)

    if len(annot_lists) < 2:
        return None, None

    def valid_intervals(items):
        return [
            x for x in items
            if isinstance(x, dict)
            and "timestamp" in x
            and isinstance(x["timestamp"], dict)
            and x["timestamp"].get("start") is not None
            and x["timestamp"].get("end") is not None
        ]

    return valid_intervals(annot_lists[0]), valid_intervals(annot_lists[1])


def get_interval_frame_info(row, video_id, images_dir, timestamp):
    """
    Frame lookup uses only raw annotation interval.
    No transcript buffer is used here.
    """
    start = timestamp.get("start")
    end = timestamp.get("end")
    frame_rate = row.get("framerate", None)

    info = {
        "frames_dir": os.path.join(images_dir, str(video_id)),
        "framerate": frame_rate,
        "target_start_frame": None,
        "target_end_frame": None,
        "actual_start_frame": None,
        "actual_end_frame": None,
        "num_frames_in_interval": 0,
    }

    if start is None or end is None:
        return info

    if frame_rate in [None, "", 0] or (isinstance(frame_rate, float) and math.isnan(frame_rate)):
        return info

    frames_dir = info["frames_dir"]
    if not os.path.isdir(frames_dir):
        return info

    target_start_frame = float(start) * float(frame_rate)
    target_end_frame = float(end) * float(frame_rate)

    info["target_start_frame"] = target_start_frame
    info["target_end_frame"] = target_end_frame

    try:
        img_path_list = get_frame_paths(frames_dir, target_start_frame, target_end_frame)
    except Exception:
        return info

    info["num_frames_in_interval"] = len(img_path_list)

    if img_path_list:
        first_name = os.path.splitext(os.path.basename(img_path_list[0]))[0]
        last_name = os.path.splitext(os.path.basename(img_path_list[-1]))[0]
        try:
            info["actual_start_frame"] = int(first_name)
            info["actual_end_frame"] = int(last_name)
        except ValueError:
            pass

    return info


def collect_clean_detection_agreed_intervals(df, data_path, images_dir, transcript_level, buffer_seconds):
    kept_records = []
    removed_no_frame = []

    for _, row in df.iterrows():
        intervals1, intervals2 = normalize_annotations(row)
        if intervals1 is None or intervals2 is None:
            continue

        overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)
        video_id = get_video_id(row)
        full_transcript = get_full_transcript(row)

        for group in overlapping_groups:
            if len(group) != 2:
                continue

            sample_a = group[0]
            sample_b = group[1]

            # Cleaning uses only detection agreement.
            if sample_a.get("error") != sample_b.get("error"):
                continue

            timestamp = sample_a["timestamp"]

            frame_info = get_interval_frame_info(
                row=row,
                video_id=video_id,
                images_dir=images_dir,
                timestamp=timestamp,
            )

            record = {
                "video_id": video_id,
                "timestamp": timestamp,
                "detection_agreement": True,
                "error": sample_a.get("error"),
                "annotator_a": sample_a,
                "annotator_b": sample_b,
                **frame_info,
            }

            if record["num_frames_in_interval"] == 0:
                removed_no_frame.append(record)
                continue

            transcript = get_transcript(
                timestamp,
                full_transcript,
                session_name=video_id,
                data_path=data_path,
                mode=transcript_level,
                buffer_seconds=buffer_seconds,
            )

            record["transcript"] = transcript
            record["has_transcript"] = is_non_empty_text(transcript)

            kept_records.append(record)

    return kept_records, removed_no_frame


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Clean detection-agreed intervals.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--transcript_level", type=str, default="exact", choices=["turn", "exact"])
    parser.add_argument("--buffer_seconds", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default="./clean_preprocess/cleaned_intervals")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)

    kept_records, removed_no_frame = collect_clean_detection_agreed_intervals(
        df=df,
        data_path=args.data_path,
        images_dir=args.images_dir,
        transcript_level=args.transcript_level,
        buffer_seconds=args.buffer_seconds,
    )

    no_transcript_records = [rec for rec in kept_records if not rec["has_transcript"]]

    cleaned_json_path = os.path.join(
        args.output_dir,
        f"{args.data_name}_detection_agreed_cleaned_intervals.json",
    )
    no_transcript_json_path = os.path.join(
        args.output_dir,
        f"{args.data_name}_detection_agreed_no_transcript_after_frame_cleaning.json",
    )
    summary_json_path = os.path.join(
        args.output_dir,
        f"{args.data_name}_detection_agreed_cleaning_summary.json",
    )

    save_json(cleaned_json_path, kept_records)
    save_json(no_transcript_json_path, no_transcript_records)
    save_json(
        summary_json_path,
        {
            "data_name": args.data_name,
            "transcript_level": args.transcript_level,
            "buffer_seconds": args.buffer_seconds,
            "num_kept_after_frame_cleaning": len(kept_records),
            "num_removed_no_frames": len(removed_no_frame),
            "num_no_transcript_after_frame_cleaning": len(no_transcript_records),
        },
    )

    print(f"[INFO] Saved cleaned intervals JSON: {cleaned_json_path}")
    print(f"[INFO] Saved no-transcript JSON: {no_transcript_json_path}")
    print(f"[INFO] Saved cleaning summary JSON: {summary_json_path}")
    print(f"[INFO] Kept after frame cleaning: {len(kept_records)}")
    print(f"[INFO] Removed due to no frames: {len(removed_no_frame)}")
    print(f"[INFO] No transcript after frame cleaning: {len(no_transcript_records)}")


if __name__ == "__main__":
    main()