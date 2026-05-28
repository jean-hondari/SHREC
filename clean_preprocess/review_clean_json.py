#!/usr/bin/env python3
import argparse
import json
import os
from copy import deepcopy


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_output_path(input_path, output_dir):
    base_name = os.path.basename(input_path)
    return os.path.join(output_dir, base_name)


def get_merged_source(sample):
    """
    Merge source labels from annotator_a and annotator_b using OR logic.

    Expected format:
        "source": {
            "Verbal": false,
            "Non-Verbal": true
        }
    """
    source_a = sample.get("annotator_a", {}).get("source", {})
    source_b = sample.get("annotator_b", {}).get("source", {})

    return {
        "Verbal": bool(source_a.get("Verbal", False) or source_b.get("Verbal", False)),
        "Non-Verbal": bool(source_a.get("Non-Verbal", False) or source_b.get("Non-Verbal", False)),
    }


def sample_matches_mode(sample, mode):
    if mode == "all":
        return True

    merged_source = get_merged_source(sample)

    if mode == "verbal":
        return merged_source["Verbal"]

    if mode == "non-verbal":
        return merged_source["Non-Verbal"]

    return False


def print_sample(sample, idx, total):
    print("\n" + "=" * 100)
    print(f"Sample {idx + 1}/{total}")
    print("=" * 100)

    print(f"video_id: {sample.get('video_id')}")
    print(f"timestamp: {sample.get('timestamp')}")
    print(f"error: {sample.get('error')}")
    print(f"status: {sample.get('status', '<unset>')}")

    print("\nMerged source (OR across annotators):")
    print(json.dumps(get_merged_source(sample), indent=2))

    print("\nAnnotator A source:")
    print(json.dumps(sample.get('annotator_a', {}).get('source', {}), indent=2))

    print("\nAnnotator B source:")
    print(json.dumps(sample.get('annotator_b', {}).get('source', {}), indent=2))

    print("\nAnnotator A rationale:")
    print(sample.get("annotator_a", {}).get("rationale", ""))

    print("\nAnnotator B rationale:")
    print(sample.get("annotator_b", {}).get("rationale", ""))

    print("\nTranscript:")
    print(sample.get("transcript", ""))


def prompt_main_action():
    while True:
        value = input("\nSelect action: [c] confirm, [r] reject, [n] next, [q] quit > ").strip().lower()
        if value in {"c", "r", "n", "q"}:
            return value
        print("Invalid input. Please select one of: c, r, n, q.")


def prompt_save_confirmation(action_name):
    while True:
        value = input(f"Really save '{action_name}'? Enter [y] to save > ").strip().lower()
        if value == "y":
            return True
        print("Input not accepted. Enter 'y' to save.")


def build_filtered_indices(data, mode):
    return [i for i, sample in enumerate(data) if sample_matches_mode(sample, mode)]


def first_unreviewed_index(data, filtered_indices):
    for idx in filtered_indices:
        if "status" not in data[idx] or data[idx]["status"] in [None, ""]:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(description="Review cleaned JSON samples and set status.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to cleaned JSON file.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "verbal", "non-verbal"],
        help="Filter by merged source label.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./clean_preprocess/reviewed_json",
        help="Directory to save reviewed JSON.",
    )

    args = parser.parse_args()

    original_data = load_json(args.input_json)
    output_path = get_output_path(args.input_json, args.output_dir)

    if os.path.exists(output_path):
        print(f"[INFO] Found existing reviewed file. Resuming from: {output_path}")
        data = load_json(output_path)
    else:
        data = deepcopy(original_data)
        save_json(output_path, data)
        print(f"[INFO] Created review file: {output_path}")

    filtered_indices = build_filtered_indices(data, args.mode)

    if not filtered_indices:
        print("[INFO] No samples matched the selected mode.")
        return

    first_idx = first_unreviewed_index(data, filtered_indices)
    if first_idx is None:
        print("[INFO] All matching samples already have status set.")
        return

    current_pos = filtered_indices.index(first_idx)

    while current_pos < len(filtered_indices):
        real_idx = filtered_indices[current_pos]
        sample = data[real_idx]

        print_sample(sample, current_pos, len(filtered_indices))
        action = prompt_main_action()

        if action == "q":
            save_json(output_path, data)
            print(f"[INFO] Saved and quit: {output_path}")
            return

        if action == "n":
            current_pos += 1
            continue

        if action == "c":
            prompt_save_confirmation("confirm")
            data[real_idx]["status"] = "c"
            save_json(output_path, data)
            print("[INFO] Saved status = c")
            current_pos += 1
            continue

        if action == "r":
            prompt_save_confirmation("reject")
            data[real_idx]["status"] = "r"
            save_json(output_path, data)
            print("[INFO] Saved status = r")
            current_pos += 1
            continue

    save_json(output_path, data)
    print(f"[INFO] Review complete. Saved file: {output_path}")


if __name__ == "__main__":
    main()