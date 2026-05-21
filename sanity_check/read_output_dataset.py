#!/usr/bin/env python3
import argparse
import pickle
import sys
import os
import random
from pprint import pprint


DISTRACTOR_KEYS = {
    "other_reason_list",
    "other_recovery_list",
    "other_transcript_list",
    "other_transcript_user_list",
    "other_transcript_agent_list",
    "other_id_list",
    "other_timestamp_list",
}


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def split_transcript_lines(sample):
    transcript = None

    if "transcript" in sample and isinstance(sample["transcript"], str):
        transcript = sample["transcript"]
    elif "transcription" in sample and isinstance(sample["transcription"], str):
        transcript = sample["transcription"]

    if not transcript:
        return None, []

    lines = transcript.splitlines()
    return transcript, lines


def sample_distractor_list(value, max_items=3):
    if not isinstance(value, list):
        return value, None

    total = len(value)
    if total <= max_items:
        return value, total

    sampled = random.sample(value, max_items)
    return sampled, total


def print_field(key, value):
    print(f"\n- {key}:")

    if key in DISTRACTOR_KEYS and isinstance(value, list):
        sampled_value, total = sample_distractor_list(value, max_items=3)
        print(f"[showing {len(sampled_value)} random item(s) out of {total}]")
        pprint(sampled_value, sort_dicts=False)
    else:
        pprint(value, sort_dicts=False)


def print_header(sample_idx, total, sample):
    print("\n" + "=" * 100)
    print(f"Sample {sample_idx + 1}/{total}")
    print("=" * 100)

    video_id = sample.get("video_id", sample.get("id", "<missing>"))
    print(f"video_id: {video_id}")

    timestamp = sample.get("timestamp")
    if isinstance(timestamp, dict):
        print(f"timestamp.start: {timestamp.get('start')}")
        print(f"timestamp.end:   {timestamp.get('end')}")
    else:
        print(f"timestamp: {timestamp}")


def print_ground_truth(sample):
    print("\n[Ground Truth / Annotation Fields]")

    preferred_order = [
        "video_id",
        "id",
        "timestamp",
        "error",
        "attribute",
        "tier2",
        "rationale",
        "correction",
        "other_reason_list",
        "other_recovery_list",
        "other_transcript_list",
        "other_transcript_user_list",
        "other_transcript_agent_list",
        "other_id_list",
        "other_timestamp_list",
        "transcript_user",
        "transcript_agent",
    ]

    printed = set()

    for key in preferred_order:
        if key in sample:
            print_field(key, sample[key])
            printed.add(key)

    for key, value in sample.items():
        if key in printed:
            continue
        if key in ("transcript", "transcription"):
            continue
        print_field(key, value)


def transcript_viewer(sample):
    transcript, lines = split_transcript_lines(sample)

    if transcript is None:
        print("\n[No transcript/transcription field found for this sample.]")
        return

    print("\n[Transcript Viewer]")
    print("Controls:")
    print("  <space> : show next line")
    print("  a       : show all remaining lines")
    print("  b       : back to sample menu")
    print("  q       : quit program")

    if len(lines) == 0:
        print("\n[Transcript exists but has no lines.]")
        return

    i = 0
    while i < len(lines):
        cmd = input("> ").rstrip("\n")

        if cmd == "q":
            print("Quitting.")
            sys.exit(0)
        elif cmd == "b":
            return
        elif cmd == "a":
            print()
            while i < len(lines):
                print(f"{i + 1:04d}: {lines[i]}")
                i += 1
            print("\n[End of transcript]")
            return
        elif cmd == " " or cmd == "":
            print(f"{i + 1:04d}: {lines[i]}")
            i += 1
            if i == len(lines):
                print("\n[End of transcript]")
                return
        else:
            print("Unknown command. Use <space>, Enter, a, b, or q.")


def sample_menu(sample_idx, total, sample):
    while True:
        print_header(sample_idx, total, sample)
        print_ground_truth(sample)

        transcript_name = None
        if "transcript" in sample:
            transcript_name = "transcript"
        elif "transcription" in sample:
            transcript_name = "transcription"

        print("\n[Options]")
        if transcript_name:
            print(f"  t : inspect {transcript_name}")
        print("  n : next sample")
        print("  q : quit")

        cmd = input("> ").strip().lower()

        if cmd == "q":
            print("Quitting.")
            sys.exit(0)
        elif cmd == "n":
            return "next"
        elif cmd == "t" and transcript_name:
            transcript_viewer(sample)
        else:
            print("Unknown command.")


def main():
    parser = argparse.ArgumentParser(description="Inspect SHREC pickle samples interactively")
    parser.add_argument("pickle_path", help="Path to .pickle file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for distractor sampling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(args.pickle_path):
        print(f"Error: file does not exist: {args.pickle_path}")
        sys.exit(1)

    data = load_pickle(args.pickle_path)

    if not isinstance(data, list):
        print(f"Loaded object is not a list. Got: {type(data)}")
        sys.exit(1)

    print(f"Loaded {len(data)} samples from: {args.pickle_path}")

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            print("\n" + "=" * 100)
            print(f"Sample {idx + 1}/{len(data)} is not a dict. Raw value:")
            pprint(sample)
            print("\nPress Enter for next sample, or q to quit.")
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            continue

        action = sample_menu(idx, len(data), sample)
        if action == "next":
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()