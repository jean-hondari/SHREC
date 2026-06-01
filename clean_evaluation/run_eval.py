import argparse
import json
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from answer_extraction import extract_answer_choice
from frame_sampling import (
    analyze_interval_frames,
    build_sample_index_to_paths,
    group_irregular_examples,
    print_frame_analysis,
    prompt_interval_seconds,
    summarize_frame_analysis,
)
from model_adapters import normalize_model_family, run_model
from path_utils import build_structured_output_dir, resolve_output_root
from prompt_builder import (
    build_attribute_prompt,
    build_correction_prompt,
    build_error_vs_competence_prompt,
    build_rationale_competence_prompt,
    build_rationale_error_prompt,
)


def load_dataset(data_path: str, task_type: str) -> List[dict]:
    dataset_path = os.path.join(data_path, task_type)
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


def load_video_to_fps(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        video_id = str(row.get("file_name", row.get("video_id", "")))
        fps = row.get("framerate", None)
        if video_id:
            mapping[video_id] = fps
    return mapping


def get_transcript(sample: dict) -> str:
    for key in ["transcript", "transcription", "conversation", "conversation_history", "text"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_ground_truth_label(sample: dict):
    if "label" in sample:
        return sample["label"]
    if "answer" in sample:
        return sample["answer"]
    if "error" in sample and isinstance(sample["error"], bool):
        return sample["error"]
    if "attribute" in sample:
        return sample["attribute"]
    if "rationale" in sample:
        return sample["rationale"]
    if "correction" in sample:
        return sample["correction"]
    return None


def make_run_config(args, interval_sec: float, model_family: str, output_dir: Path) -> dict:
    return {
        "task_type": args.task_type,
        "model": args.model,
        "model_family": model_family,
        "seed": args.seed,
        "temperature": args.temperature,
        "interval_sec": interval_sec,
        "data_path": str(Path(args.data_path).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
        "csv_path": str(Path(args.csv_path).resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def run_config_matches(existing: dict, current: dict) -> bool:
    keys = ["task_type", "model", "model_family", "seed", "temperature", "interval_sec"]
    return all(existing.get(k) == current.get(k) for k in keys)


def default_output_path(output_dir: Path, config: dict) -> Path:
    stem = (
        f"{config['task_type'].replace('.pickle', '')}"
        f"__{config['model']}"
        f"__seed{config['seed']}"
        f"__temp{config['temperature']}"
        f"__interval{config['interval_sec']}"
    )
    return output_dir / f"{stem}.json"


def load_existing_run(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "run_config" in data and "results" in data:
            return data
    except Exception:
        return None
    return None


def summarize_error_results(results: List[dict]) -> Dict[str, int]:
    counter = Counter()
    for item in results:
        err = item.get("error")
        if err:
            key = f"{err.get('category', 'unknown')} | {err.get('type', 'UnknownError')} | {err.get('message', '')}"
            counter[key] += 1
    return dict(counter)


def prompt_retry_errors(error_summary: Dict[str, int]) -> bool:
    if not error_summary:
        return False

    print("\nFound previous errored samples:")
    for msg, count in error_summary.items():
        print(f"- {count} x {msg}")

    value = input("\nResume errored samples too? [y/N]: ").strip().lower()
    return value == "y"


def choose_resume_or_new(output_dir: Path, run_config: dict) -> Path:
    candidate = default_output_path(output_dir, run_config)
    existing = load_existing_run(candidate)

    if existing is None:
        return candidate

    if run_config_matches(existing["run_config"], run_config):
        print(f"[INFO] Matching previous run found: {candidate}")
        return candidate

    suffix = 2
    while True:
        alt = candidate.with_name(candidate.stem + f"__run{suffix}" + candidate.suffix)
        if not alt.exists():
            print(f"[INFO] Existing file has different config. Saving new run to: {alt}")
            return alt
        suffix += 1


def normalize_result_key(sample: dict, sample_index: int) -> str:
    video_id = str(sample.get("video_id", ""))
    ts = sample.get("timestamp", {}) or {}
    start = ts.get("start")
    end = ts.get("end")
    return f"{sample_index}::{video_id}::{start}::{end}"


def save_run(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def shuffle_single_correct_choices(
    distractors: List[str],
    correct_text: str,
    rng: random.Random,
    total_choices: int = 5,
) -> Tuple[List[str], str]:
    """
    Returns:
      displayed_choices: list[str]
      ground_truth_choice: '(k)'
    """
    distractors = [x for x in distractors if x is not None and str(x).strip()]
    deduped = []
    seen = set()

    for item in distractors:
        norm = str(item).strip()
        if norm == str(correct_text).strip():
            continue
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)

    displayed_choices = deduped[: total_choices - 1]
    displayed_choices.append(str(correct_text).strip())
    rng.shuffle(displayed_choices)

    gt_index = displayed_choices.index(str(correct_text).strip()) + 1
    ground_truth_choice = f"({gt_index})"
    return displayed_choices, ground_truth_choice


def normalize_attribute_ground_truth(label) -> Optional[str]:
    """
    Keep this lightweight: convert dict/list into a canonical comma-separated choice string.
    No correctness scoring yet; this is just stored metadata.
    """
    if label is None:
        return None

    if isinstance(label, str):
        return label

    if isinstance(label, list):
        vals = [str(x).strip() for x in label if str(x).strip()]
        return ",".join(vals) if vals else None

    if isinstance(label, dict):
        active = [k for k, v in label.items() if v]
        return ",".join(active) if active else None

    return str(label)


def build_prompt_and_ground_truth(task_type: str, sample: dict, transcript: str, rng: random.Random):
    lower = task_type.lower()

    if "error_vs_competence" in lower:
        prompt = build_error_vs_competence_prompt(task_type, transcript)
        gt_raw = sample.get("error")
        if gt_raw is True:
            ground_truth_choice = "B"
        elif gt_raw is False:
            ground_truth_choice = "A"
        else:
            ground_truth_choice = None

        return {
            "prompt": prompt,
            "ground_truth_label": gt_raw,
            "ground_truth_choice": ground_truth_choice,
            "presented_choices": {
                "A": "Social Competence",
                "B": "Social Error",
            },
        }

    if "attribute" in lower:
        prompt = build_attribute_prompt(task_type, transcript, sample)
        gt_raw = sample.get("attribute")
        gt_choice = normalize_attribute_ground_truth(gt_raw)

        return {
            "prompt": prompt,
            "ground_truth_label": gt_raw,
            "ground_truth_choice": gt_choice,
            "presented_choices": {
                "A": "Emotions",
                "B": "Engagement",
                "C": "Conversational Mechanics",
                "D": "Knowledge State",
                "E": "User Intention",
                "F": "Social Context and Relationships",
                "G": "Social Norms and Routines",
            },
        }

    if "rationale_error" in lower:
        correct_text = sample.get("rationale")
        distractors = list(sample.get("other_reason_list", []))
        displayed_choices, ground_truth_choice = shuffle_single_correct_choices(
            distractors=distractors,
            correct_text=correct_text,
            rng=rng,
            total_choices=5,
        )
        prompt = build_rationale_error_prompt(task_type, transcript, displayed_choices)

        return {
            "prompt": prompt,
            "ground_truth_label": correct_text,
            "ground_truth_choice": ground_truth_choice,
            "presented_choices": {
                f"({i+1})": displayed_choices[i] for i in range(len(displayed_choices))
            },
        }

    if "rationale_competence" in lower:
        correct_text = sample.get("rationale")
        distractors = list(sample.get("other_reason_list", []))
        displayed_choices, ground_truth_choice = shuffle_single_correct_choices(
            distractors=distractors,
            correct_text=correct_text,
            rng=rng,
            total_choices=5,
        )
        prompt = build_rationale_competence_prompt(task_type, transcript, displayed_choices)

        return {
            "prompt": prompt,
            "ground_truth_label": correct_text,
            "ground_truth_choice": ground_truth_choice,
            "presented_choices": {
                f"({i+1})": displayed_choices[i] for i in range(len(displayed_choices))
            },
        }

    if "correction" in lower:
        correct_text = sample.get("correction")
        distractors = list(sample.get("other_recovery_list", []))
        displayed_choices, ground_truth_choice = shuffle_single_correct_choices(
            distractors=distractors,
            correct_text=correct_text,
            rng=rng,
            total_choices=5,
        )
        prompt = build_correction_prompt(task_type, transcript, displayed_choices)

        return {
            "prompt": prompt,
            "ground_truth_label": correct_text,
            "ground_truth_choice": ground_truth_choice,
            "presented_choices": {
                f"({i+1})": displayed_choices[i] for i in range(len(displayed_choices))
            },
        }

    # fallback
    prompt = f"Conversation History:\n{transcript}"
    return {
        "prompt": prompt,
        "ground_truth_label": get_ground_truth_label(sample),
        "ground_truth_choice": None,
        "presented_choices": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Resumable evaluation runner with frame interval sampling")
    parser.add_argument("--task_type", type=str, required=True, help="pickle filename from clean_preprocess/output_datasets")
    parser.add_argument("--model", type=str, required=True, choices=["gpt-4o-mini", "gpt-4o-mini-vision", "internvl", "llama"])
    parser.add_argument("--data_path", type=str, default="./clean_preprocess/output_datasets")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    dataset = load_dataset(args.data_path, args.task_type)
    if args.shuffle:
        random.shuffle(dataset)

    model_family = normalize_model_family(args.model)
    video_to_fps = load_video_to_fps(args.csv_path)

    infos = analyze_interval_frames(
        dataset=dataset,
        images_dir=args.images_dir,
        video_to_fps=video_to_fps,
    )
    summary = summarize_frame_analysis(infos)
    print_frame_analysis(summary)

    irregular_infos = [x for x in infos if not x.regular_interval]
    if irregular_infos:
        grouped = group_irregular_examples(irregular_infos)
        print("\n[WARNING] Irregular frame intervals detected.")
        for reason, examples in grouped.items():
            print(f"- {reason}: {len(examples)} example(s) shown")
            for ex in examples:
                print(
                    f"  sample_index={ex['sample_index']} "
                    f"video_id={ex['video_id']} "
                    f"num_frames={ex['num_frames']} "
                    f"min_delta_frames={ex['min_delta_frames']} "
                    f"max_delta_frames={ex['max_delta_frames']}"
                )
        print("\nProceeding anyway, but sampling will be based on actual frame timestamps derived from filename/fps.")

    interval_sec = prompt_interval_seconds()
    sample_index_to_paths = build_sample_index_to_paths(infos, interval_sec)

    output_root = resolve_output_root(args.output_dir)
    output_dir = build_structured_output_dir(output_root, args.task_type, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = make_run_config(args, interval_sec, model_family, output_dir)

    print(f"[INFO] Input data_path: {Path(args.data_path).resolve()}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Structured output dir: {output_dir}")

    output_path = choose_resume_or_new(output_dir, run_config)
    print(f"[INFO] Output file: {output_path}")

    existing = load_existing_run(output_path)

    if existing is None:
        payload = {
            "run_config": run_config,
            "frame_analysis_summary": summary,
            "results": [],
        }
    else:
        payload = existing
        print("\n[INFO] Resuming existing run with config:")
        for k, v in payload["run_config"].items():
            if k in {"task_type", "model", "seed", "temperature", "interval_sec"}:
                print(f"- {k}: {v}")

    result_map = {}
    for item in payload["results"]:
        key = item.get("sample_key")
        if key:
            result_map[key] = item

    error_summary = summarize_error_results(payload["results"])
    retry_errors = prompt_retry_errors(error_summary)

    pending_indices = []
    for i, sample in enumerate(dataset):
        key = normalize_result_key(sample, i)
        existing_item = result_map.get(key)
        if existing_item is None:
            pending_indices.append(i)
        elif existing_item.get("error") and retry_errors:
            pending_indices.append(i)

    print(f"\n[INFO] Total dataset size: {len(dataset)}")
    print(f"[INFO] Pending samples in this run: {len(pending_indices)}")

    progress = tqdm(pending_indices, total=len(pending_indices), desc="Evaluating")

    for i in progress:
        sample = dataset[i]
        transcript = get_transcript(sample)

        task_payload = build_prompt_and_ground_truth(
            task_type=args.task_type,
            sample=sample,
            transcript=transcript,
            rng=rng,
        )

        prompt = task_payload["prompt"]
        ground_truth_label = task_payload["ground_truth_label"]
        ground_truth_choice = task_payload["ground_truth_choice"]
        presented_choices = task_payload["presented_choices"]

        image_paths = sample_index_to_paths.get(i, [])

        response, error_payload = run_model(
            model_name=args.model,
            prompt=prompt,
            transcript=transcript,
            image_paths=image_paths if "vlm" in model_family else [],
            temperature=args.temperature,
            seed=args.seed,
        )

        answer_choice = extract_answer_choice(args.task_type, response)

        result_item = {
            "sample_index": i,
            "sample_key": normalize_result_key(sample, i),
            "video_id": sample.get("video_id"),
            "timestamp": sample.get("timestamp"),
            "task_type": args.task_type,
            "model": args.model,
            "seed": args.seed,
            "temperature": args.temperature,
            "interval_sec": interval_sec,
            "prompt": prompt,
            "transcript_used": transcript,
            "image_paths_used": image_paths if "vlm" in model_family else [],
            "num_frames_used": len(image_paths) if "vlm" in model_family else 0,
            "presented_choices": presented_choices,
            "ground_truth_label": ground_truth_label,
            "ground_truth_choice": ground_truth_choice,
            "response": response,
            "answer_choice": answer_choice,
            "error": error_payload,
        }

        result_map[result_item["sample_key"]] = result_item
        payload["results"] = sorted(result_map.values(), key=lambda x: x["sample_index"])
        save_run(output_path, payload)

        completed = sum(1 for r in payload["results"] if not r.get("error"))
        errored = sum(1 for r in payload["results"] if r.get("error"))
        progress.set_postfix(done=completed, errors=errored, left=len(dataset) - len(payload["results"]))

    error_count = sum(1 for r in payload["results"] if r.get("error"))

    print("\n[Final statistics]")
    print(f"- total results saved: {len(payload['results'])}")
    print(f"- errored: {error_count}")
    print(f"- saved to: {output_path}")


if __name__ == "__main__":
    main()