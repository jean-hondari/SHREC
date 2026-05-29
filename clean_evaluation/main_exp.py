import os
import json
import pickle
import random
import argparse
from pathlib import Path

from tqdm import tqdm

from exp_llm import build_base_prompt, call_llm, save_results as save_llm_results
from exp_vlm import call_vlm, save_results as save_vlm_results
from utils import get_frame_paths

random.seed(42)


def load_dataset(data_path: str, task_type: str):
    dataset_path = os.path.join(data_path, task_type)
    with open(dataset_path, "rb") as pkl_file:
        processed_dataset = pickle.load(pkl_file)
    return processed_dataset


def load_existing_results(output_file: Path):
    if output_file.exists():
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
    return []


def build_question(sample, base_prompt: str):
    if "question" in sample and sample["question"]:
        return f"{base_prompt}\n\n{sample['question']}"
    return base_prompt


def get_sample_transcript(sample):
    for key in ["transcript", "conversation", "conversation_history", "transcription", "text"]:
        if key in sample and sample[key]:
            return sample[key]
    return ""


def infer_frame_range(sample):
    start_frame = sample.get("start_frame", sample.get("frame_start"))
    end_frame = sample.get("end_frame", sample.get("frame_end"))

    if start_frame is not None and end_frame is not None:
        return start_frame, end_frame

    if "timestamp" in sample and isinstance(sample["timestamp"], dict):
        return sample["timestamp"].get("start"), sample["timestamp"].get("end")

    return None, None


def extract_image_paths(sample, images_dir: str):
    if "image_paths" in sample and isinstance(sample["image_paths"], list):
        return sample["image_paths"]

    if "img_paths" in sample and isinstance(sample["img_paths"], list):
        return sample["img_paths"]

    frame_dir = sample.get("frame_dir", images_dir)
    start_frame = sample.get("start_frame", sample.get("frame_start"))
    end_frame = sample.get("end_frame", sample.get("frame_end"))

    if frame_dir and start_frame is not None and end_frame is not None and os.path.isdir(frame_dir):
        return get_frame_paths(frame_dir, start_frame, end_frame)

    return []


def present_human_sample(idx, total, sample, prompt, transcript, image_paths):
    start_frame, end_frame = infer_frame_range(sample)

    print("\n" + "=" * 80)
    print(f"Sample {idx + 1}/{total}")
    print("=" * 80)

    print("\nPROMPT:")
    print(prompt)

    print("\nTRANSCRIPT:")
    print(transcript if transcript else "[No transcript available]")

    print("\nFRAME INFORMATION:")
    print(f"Number of frames: {len(image_paths)}")
    print(f"Start frame: {start_frame}")
    print(f"End frame: {end_frame}")

    if "label" in sample:
        print("\nLABEL:")
        print(sample["label"])

    print("\nEnter your response:")
    return input("> ")


def run_human_mode(dataset, args, base_prompt, output_file):
    results = load_existing_results(output_file)

    for idx, sample in enumerate(dataset[len(results):], start=len(results)):
        transcript = get_sample_transcript(sample)
        prompt = build_question(sample, base_prompt)
        image_paths = extract_image_paths(sample, args.images_dir)

        response = present_human_sample(idx, len(dataset), sample, prompt, transcript, image_paths)

        result = dict(sample)
        result["prompt"] = prompt
        result["transcript_used"] = transcript
        result["num_frames"] = len(image_paths)
        result["response"] = response
        results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved human evaluation results to {output_file}")


def run_llm_mode(dataset, args, base_prompt, output_file):
    results = load_existing_results(output_file)

    pending = dataset[len(results):]
    for sample in tqdm(pending, total=len(pending), desc="Running LLM evaluation"):
        transcript = get_sample_transcript(sample)
        prompt = build_question(sample, base_prompt)

        response = call_llm(
            model_name=args.model,
            prompt=prompt,
            transcript=transcript,
        )

        result = dict(sample)
        result["prompt"] = prompt
        result["transcript_used"] = transcript
        result["response"] = response
        results.append(result)

        save_llm_results(results, str(output_file))

    print(f"Saved LLM evaluation results to {output_file}")


def run_vlm_mode(dataset, args, base_prompt, output_file):
    results = load_existing_results(output_file)

    pending = dataset[len(results):]
    for sample in tqdm(pending, total=len(pending), desc="Running VLM evaluation"):
        transcript = get_sample_transcript(sample)
        prompt = build_question(sample, base_prompt)
        image_paths = extract_image_paths(sample, args.images_dir)

        response = call_vlm(
            model_name=args.model,
            prompt=prompt,
            transcript=transcript,
            image_paths=image_paths,
        )

        result = dict(sample)
        result["prompt"] = prompt
        result["transcript_used"] = transcript
        result["image_paths_used"] = image_paths
        result["num_frames"] = len(image_paths)
        result["response"] = response
        results.append(result)

        save_vlm_results(results, str(output_file))

    print(f"Saved VLM evaluation results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Experiment runner for human, vlm, and llm modes")
    parser.add_argument("--mode", type=str, choices=["human", "vlm", "llm"], required=True)
    parser.add_argument("--task_type", type=str, required=True, help="pickle filename from clean_preprocess")
    parser.add_argument("--model", type=str, default="gemini-pro-vision")
    parser.add_argument("--data_path", type=str, default="./clean_preprocess")
    parser.add_argument("--images_dir", type=str, default="../shrec_empathic")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    dataset = load_dataset(args.data_path, args.task_type)
    random.shuffle(dataset)

    base_prompt = build_base_prompt(args.task_type, args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.mode}_{args.model}_{args.task_type.replace('.pickle', '')}.json"

    if args.mode == "human":
        run_human_mode(dataset, args, base_prompt, output_file)
    elif args.mode == "llm":
        run_llm_mode(dataset, args, base_prompt, output_file)
    elif args.mode == "vlm":
        run_vlm_mode(dataset, args, base_prompt, output_file)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()