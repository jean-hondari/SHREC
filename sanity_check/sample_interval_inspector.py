#!/usr/bin/env python3
import argparse
import ast
import math
import os
import random

import pandas as pd

from utils_exact import (
    find_overlapping_interval_groups_pair,
    get_transcript,
    get_frame_paths,
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


def get_video_id(row):
    for key in ["file_name", "video_id", "id"]:
        if key in row and pd.notna(row[key]):
            return row[key]
    return "<unknown>"


def get_full_transcript(row):
    for key in ["transcript", "transcription"]:
        if key in row and isinstance(row[key], str):
            return row[key]
    return ""


def format_attribute(attr):
    if isinstance(attr, dict):
        active = [k for k, v in attr.items() if v]
        return active if active else attr
    return attr


def get_interval_frame_info(row, video_id, images_dir, start, end):
    frame_rate = row.get("framerate", None)

    info = {
        "framerate": frame_rate,
        "target_start_frame": None,
        "target_end_frame": None,
        "actual_start_frame": None,
        "actual_end_frame": None,
        "num_frames_in_interval": 0,
        "frames_dir": os.path.join(images_dir, str(video_id)),
    }

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


def build_record(row, sample, data_path, images_dir, transcript_level="exact", buffer_seconds=5.0):
    video_id = get_video_id(row)
    timestamp = sample.get("timestamp", {})
    start = timestamp.get("start")
    end = timestamp.get("end")
    if start is None or end is None:
        return None

    full_transcript = get_full_transcript(row)
    transcript = get_transcript(
        timestamp,
        full_transcript,
        session_name=video_id,
        data_path=data_path,
        mode=transcript_level,
        buffer_seconds=buffer_seconds,
    )

    frame_info = get_interval_frame_info(
        row=row,
        video_id=video_id,
        images_dir=images_dir,
        start=start,
        end=end,
    )

    return {
        "video_id": video_id,
        "start_time": start,
        "end_time": end,
        "transcript": transcript,
        "detection_error_type": sample.get("error"),
        "attribute_error_type": format_attribute(sample.get("attribute")),
        "rationale": sample.get("rationale", ""),
        **frame_info,
    }


def collect_records(df, data_path, images_dir, mode, transcript_level="exact", buffer_seconds=5.0):
    records = []

    for _, row in df.iterrows():
        intervals1, intervals2 = normalize_annotations(row)
        if intervals1 is None or intervals2 is None:
            continue

        overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

        for group in overlapping_groups:
            if not group:
                continue

            if mode == "agreed":
                if len(group) == 2 and group[0].get("error") == group[1].get("error"):
                    rec = build_record(
                        row,
                        group[0],
                        data_path=data_path,
                        images_dir=images_dir,
                        transcript_level=transcript_level,
                        buffer_seconds=buffer_seconds,
                    )
                    if rec is not None:
                        records.append(rec)

            elif mode == "singleton":
                if len(group) == 1:
                    rec = build_record(
                        row,
                        group[0],
                        data_path=data_path,
                        images_dir=images_dir,
                        transcript_level=transcript_level,
                        buffer_seconds=buffer_seconds,
                    )
                    if rec is not None:
                        records.append(rec)

            elif mode == "all":
                for sample in group:
                    rec = build_record(
                        row,
                        sample,
                        data_path=data_path,
                        images_dir=images_dir,
                        transcript_level=transcript_level,
                        buffer_seconds=buffer_seconds,
                    )
                    if rec is not None:
                        records.append(rec)

            else:
                raise ValueError(f"Unsupported mode: {mode}")

    return records


def print_sample(sample, sample_num=None):
    print("\n" + "=" * 100)
    if sample_num is not None:
        print(f"Random Sample #{sample_num}")
    else:
        print("Random Sample")
    print("=" * 100)

    print(f"video name: {sample['video_id']}")
    print(f"actual start time: {sample['start_time']}")
    print(f"actual end time:   {sample['end_time']}")
    print(f"target start frame (time * fps): {sample['target_start_frame']}")
    print(f"target end frame (time * fps):   {sample['target_end_frame']}")
    print(f"actual start frame in dir:       {sample['actual_start_frame']}")
    print(f"actual end frame in dir:         {sample['actual_end_frame']}")
    print(f"num frames in interval:          {sample['num_frames_in_interval']}")
    print(f"framerate:                       {sample['framerate']}")
    print(f"detection error type: {sample['detection_error_type']}")
    print(f"attribute error type: {sample['attribute_error_type']}")
    print(f"rationale: {sample['rationale']}")

    print("\n[transcript]")
    print(sample["transcript"] if sample["transcript"] else "<empty transcript>")


def interactive_loop(records):
    if not records:
        print("No valid intervals found.")
        return

    seen_indices = set()
    sample_count = 0

    while True:
        if len(seen_indices) == len(records):
            print("\nAll samples have been shown once. Resetting pool.\n")
            seen_indices.clear()

        available_indices = [i for i in range(len(records)) if i not in seen_indices]
        idx = random.choice(available_indices)
        seen_indices.add(idx)

        sample_count += 1
        print_sample(records[idx], sample_num=sample_count)

        print("\nOptions:")
        print("  n     -> next random sample")
        print("  quit  -> exit")
        cmd = input("> ").strip().lower()

        if cmd == "quit":
            print("Exiting.")
            break
        elif cmd == "n":
            continue
        else:
            print("Unknown command. Type 'n' or 'quit'.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively sample random intervals and inspect transcript + annotations + actual frame range."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="agreed",
        choices=["agreed", "singleton", "all"],
    )
    parser.add_argument(
        "--transcript_level",
        type=str,
        default="exact",
        choices=["turn", "exact"],
    )
    parser.add_argument("--buffer_seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.data_path)

    records = collect_records(
        df=df,
        data_path=args.data_path,
        images_dir=args.images_dir,
        mode=args.mode,
        transcript_level=args.transcript_level,
        buffer_seconds=args.buffer_seconds,
    )

    print(f"Loaded {len(records)} candidate intervals.")
    interactive_loop(records)


if __name__ == "__main__":
    main()