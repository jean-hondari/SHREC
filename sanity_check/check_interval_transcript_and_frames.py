#!/usr/bin/env python3
import argparse
import ast
import math
import os
from collections import defaultdict

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


def is_non_empty_text(text):
    return isinstance(text, str) and text.strip() != ""


def normalize_annotations(row):
    annotations_a = safe_literal_eval(row.get("Annotations_A", []))
    annotations_b = safe_literal_eval(row.get("Annotations_B", []))
    annotations_c = safe_literal_eval(row.get("Annotations_C", []))

    annot_lists = []
    if isinstance(annotations_a, list) and len(annotations_a) > 0:
        annot_lists.append(annotations_a)
    if isinstance(annotations_b, list) and len(annotations_b) > 0:
        annot_lists.append(annotations_b)
    if isinstance(annotations_c, list) and len(annotations_c) > 0:
        annot_lists.append(annotations_c)

    if len(annot_lists) < 2:
        return None, None

    intervals1 = [
        x for x in annot_lists[0]
        if isinstance(x, dict)
        and "timestamp" in x
        and x["timestamp"].get("start") is not None
        and x["timestamp"].get("end") is not None
    ]
    intervals2 = [
        x for x in annot_lists[1]
        if isinstance(x, dict)
        and "timestamp" in x
        and x["timestamp"].get("start") is not None
        and x["timestamp"].get("end") is not None
    ]
    return intervals1, intervals2


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


def get_frame_count_for_interval(row, video_id, images_dir, start, end):
    """
    Match main_vlm_exp.py behavior:
      frame_rate = df[df['file_name'] == video_id]['framerate'].item()
      frames_dir = os.path.join(images_dir, video_id)
      img_path_list = get_frame_paths(frames_dir, start * frame_rate, end * frame_rate)

    IMPORTANT: no transcript buffer here.
    """
    frame_rate = row.get("framerate", None)
    if frame_rate in [None, "", 0] or (isinstance(frame_rate, float) and math.isnan(frame_rate)):
        return 0

    frames_dir = os.path.join(images_dir, video_id)
    if not os.path.isdir(frames_dir):
        return 0

    try:
        start_frame = float(start) * float(frame_rate)
        end_frame = float(end) * float(frame_rate)
        img_path_list = get_frame_paths(frames_dir, start_frame, end_frame)
        return len(img_path_list)
    except Exception:
        return 0


def build_record(row, sample, transcript_level, buffer_seconds, data_path, images_dir):
    video_id = get_video_id(row)
    timestamp = sample.get("timestamp", {})
    start = timestamp.get("start")
    end = timestamp.get("end")
    if start is None or end is None:
        return None

    full_transcript = get_full_transcript(row)

    # transcript uses buffer_seconds
    transcript = get_transcript(
        timestamp,
        full_transcript,
        session_name=video_id,
        data_path=data_path,
        mode=transcript_level,
        buffer_seconds=buffer_seconds,
    )

    # frame count uses raw interval only, same style as main_vlm_exp.py
    frame_count = get_frame_count_for_interval(
        row=row,
        video_id=video_id,
        images_dir=images_dir,
        start=start,
        end=end,
    )

    return {
        "video_id": video_id,
        "start": start,
        "end": end,
        "error": sample.get("error", None),
        "rationale": sample.get("rationale", ""),
        "has_transcript": is_non_empty_text(transcript),
        "frame_count": frame_count,
    }


def summarize_records(name, records):
    total = len(records)
    transcript_non_empty = sum(r["has_transcript"] for r in records)
    frames_non_zero = sum(r["frame_count"] > 0 for r in records)
    avg_frames = (sum(r["frame_count"] for r in records) / total) if total > 0 else 0.0

    print(f"\n=== {name} ===")
    print(f"Total intervals: {total}")
    print(f"Intervals with non-empty transcript: {transcript_non_empty}")
    print(f"Intervals with non-zero frames: {frames_non_zero}")
    print(f"Average frame count: {avg_frames:.3f}")


def select_records_for_mode(row, overlapping_groups, mode, transcript_level, buffer_seconds, data_path, images_dir):
    records = []

    for group in overlapping_groups:
        if not group:
            continue

        if mode == "agreed":
            if len(group) == 2 and group[0].get("error") == group[1].get("error"):
                rec = build_record(
                    row, group[0], transcript_level, buffer_seconds, data_path, images_dir
                )
                if rec is not None:
                    records.append(rec)

        elif mode == "singleton":
            if len(group) == 1:
                rec = build_record(
                    row, group[0], transcript_level, buffer_seconds, data_path, images_dir
                )
                if rec is not None:
                    records.append(rec)

        elif mode == "rationale_repo_exact":
            sample = group[0]
            rationale = sample.get("rationale", "")
            if isinstance(rationale, str) and len(rationale) > 0:
                rec = build_record(
                    row, sample, transcript_level, buffer_seconds, data_path, images_dir
                )
                if rec is not None:
                    records.append(rec)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["agreed", "singleton", "rationale_repo_exact"],
    )
    parser.add_argument(
        "--transcript_level",
        type=str,
        default="exact",
        choices=["turn", "exact"],
    )
    parser.add_argument(
        "--buffer_seconds",
        type=float,
        default=5.0,
        help="Used only for transcript extraction.",
    )
    parser.add_argument("--print_per_video", action="store_true")

    args = parser.parse_args()
    df = pd.read_csv(args.data_path)

    all_records = []
    per_video = defaultdict(list)

    for _, row in df.iterrows():
        intervals1, intervals2 = normalize_annotations(row)
        if intervals1 is None or intervals2 is None:
            continue

        overlapping_groups = find_overlapping_interval_groups_pair(intervals1, intervals2)

        records = select_records_for_mode(
            row=row,
            overlapping_groups=overlapping_groups,
            mode=args.mode,
            transcript_level=args.transcript_level,
            buffer_seconds=args.buffer_seconds,
            data_path=args.data_path,
            images_dir=args.images_dir,
        )

        all_records.extend(records)
        for rec in records:
            per_video[rec["video_id"]].append(rec)

    summarize_records(f"Mode: {args.mode}", all_records)

    if args.print_per_video:
        for video_id in sorted(per_video.keys()):
            summarize_records(f"{video_id} [{args.mode}]", per_video[video_id])


if __name__ == "__main__":
    main()