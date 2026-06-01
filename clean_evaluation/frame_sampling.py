import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class IntervalFrameInfo:
    sample_index: int
    video_id: str
    frames_dir: str
    framerate: Optional[float]
    start_time: Optional[float]
    end_time: Optional[float]
    target_start_frame: Optional[float]
    target_end_frame: Optional[float]
    frame_paths_in_interval: List[str]
    frame_indices_in_interval: List[int]
    frame_times_in_interval: List[float]
    min_delta_frames: Optional[int]
    max_delta_frames: Optional[int]
    regular_interval: bool
    irregular_reason: Optional[str]


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def list_integer_png_frames(frames_dir: str) -> List[int]:
    if not os.path.isdir(frames_dir):
        return []

    indices = []
    for fname in os.listdir(frames_dir):
        if not fname.endswith(".png"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem.isdigit():
            indices.append(int(stem))
    indices.sort()
    return indices


def get_frame_paths_between_indices(frames_dir: str, start_frame: float, end_frame: float) -> List[str]:
    all_indices = list_integer_png_frames(frames_dir)
    selected = [idx for idx in all_indices if start_frame <= idx <= end_frame]
    return [os.path.join(frames_dir, f"{idx}.png") for idx in selected]


def extract_transcript(sample: dict) -> str:
    for key in ["transcript", "transcription", "conversation", "conversation_history", "text"]:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_label(sample: dict):
    for key in ["label", "answer", "target", "gold", "error", "attribute", "rationale", "correction"]:
        if key in sample:
            return sample[key]
    return None


def analyze_interval_frames(
    dataset: List[dict],
    images_dir: str,
    video_to_fps: Dict[str, float],
) -> List[IntervalFrameInfo]:
    analyzed = []

    for i, sample in enumerate(dataset):
        video_id = str(sample.get("video_id", ""))
        ts = sample.get("timestamp", {}) or {}
        start_time = safe_float(ts.get("start"))
        end_time = safe_float(ts.get("end"))
        fps = safe_float(video_to_fps.get(video_id))

        frames_dir = os.path.join(images_dir, video_id)

        target_start_frame = None
        target_end_frame = None
        frame_paths = []
        frame_indices = []
        frame_times = []
        min_delta_frames = None
        max_delta_frames = None
        regular = True
        irregular_reason = None

        if start_time is None or end_time is None:
            regular = False
            irregular_reason = "missing_timestamp"
        elif fps is None or fps <= 0:
            regular = False
            irregular_reason = "missing_fps"
        elif not os.path.isdir(frames_dir):
            regular = False
            irregular_reason = "missing_frames_dir"
        else:
            target_start_frame = start_time * fps
            target_end_frame = end_time * fps
            frame_paths = get_frame_paths_between_indices(frames_dir, target_start_frame, target_end_frame)
            frame_indices = [
                int(os.path.splitext(os.path.basename(path))[0])
                for path in frame_paths
            ]
            frame_times = [idx / fps for idx in frame_indices]

            if len(frame_indices) >= 2:
                deltas = [
                    frame_indices[j + 1] - frame_indices[j]
                    for j in range(len(frame_indices) - 1)
                ]
                min_delta_frames = min(deltas)
                max_delta_frames = max(deltas)
                regular = (min_delta_frames == max_delta_frames)
                if not regular:
                    irregular_reason = "irregular_frame_spacing"
            elif len(frame_indices) <= 1:
                regular = True

        analyzed.append(
            IntervalFrameInfo(
                sample_index=i,
                video_id=video_id,
                frames_dir=frames_dir,
                framerate=fps,
                start_time=start_time,
                end_time=end_time,
                target_start_frame=target_start_frame,
                target_end_frame=target_end_frame,
                frame_paths_in_interval=frame_paths,
                frame_indices_in_interval=frame_indices,
                frame_times_in_interval=frame_times,
                min_delta_frames=min_delta_frames,
                max_delta_frames=max_delta_frames,
                regular_interval=regular,
                irregular_reason=irregular_reason,
            )
        )

    return analyzed


def summarize_frame_analysis(infos: List[IntervalFrameInfo]) -> dict:
    counts = [len(info.frame_indices_in_interval) for info in infos]
    irregular = [info for info in infos if not info.regular_interval]

    summary = {
        "num_samples": len(infos),
        "min_num_frames": min(counts) if counts else 0,
        "max_num_frames": max(counts) if counts else 0,
        "num_irregular_samples": len(irregular),
        "irregular_reason_counts": dict(Counter(info.irregular_reason for info in irregular)),
    }

    fps_values = [info.framerate for info in infos if info.framerate is not None]
    if fps_values:
        summary["min_fps"] = min(fps_values)
        summary["max_fps"] = max(fps_values)
    else:
        summary["min_fps"] = None
        summary["max_fps"] = None

    approx_one_fps_counts = []
    for info in infos:
        if info.start_time is None or info.end_time is None:
            continue
        duration = max(0.0, info.end_time - info.start_time)
        approx_one_fps_counts.append(max(1, math.floor(duration) + 1))

    summary["approx_min_frames_at_1fps"] = min(approx_one_fps_counts) if approx_one_fps_counts else 0
    summary["approx_max_frames_at_1fps"] = max(approx_one_fps_counts) if approx_one_fps_counts else 0
    return summary


def print_frame_analysis(summary: dict):
    print("\n[Frame interval analysis]")
    print(f"- samples analyzed: {summary['num_samples']}")
    print(f"- min frames in annotated interval: {summary['min_num_frames']}")
    print(f"- max frames in annotated interval: {summary['max_num_frames']}")
    print(f"- irregular samples: {summary['num_irregular_samples']}")

    if summary["irregular_reason_counts"]:
        print("- irregular reason counts:")
        for reason, count in summary["irregular_reason_counts"].items():
            print(f"  - {reason}: {count}")

    print(f"- approx min frames at 1 frame/sec: {summary['approx_min_frames_at_1fps']}")
    print(f"- approx max frames at 1 frame/sec: {summary['approx_max_frames_at_1fps']}")


def prompt_interval_seconds() -> float:
    while True:
        value = input("\nSelect frame sampling interval in seconds (e.g. 1, 0.5, 2): ").strip()
        try:
            interval_sec = float(value)
            if interval_sec <= 0:
                raise ValueError
            return interval_sec
        except Exception:
            print("Invalid interval. Please enter a positive number.")


def sample_frame_paths_for_interval(
    info: IntervalFrameInfo,
    interval_sec: float,
) -> List[str]:
    if not info.frame_indices_in_interval or not info.frame_times_in_interval:
        return []

    selected = []
    next_target_time = info.start_time

    for path, t in zip(info.frame_paths_in_interval, info.frame_times_in_interval):
        if next_target_time is None:
            break
        if t + 1e-9 >= next_target_time:
            selected.append(path)
            next_target_time += interval_sec

    if not selected and info.frame_paths_in_interval:
        selected.append(info.frame_paths_in_interval[0])

    return selected


def build_sample_index_to_paths(
    infos: List[IntervalFrameInfo],
    interval_sec: float,
) -> Dict[int, List[str]]:
    mapping = {}
    for info in infos:
        mapping[info.sample_index] = sample_frame_paths_for_interval(info, interval_sec)
    return mapping


def group_irregular_examples(infos: List[IntervalFrameInfo], max_examples: int = 10) -> dict:
    grouped = defaultdict(list)
    for info in infos:
        if not info.regular_interval:
            grouped[info.irregular_reason].append(info)

    out = {}
    for reason, items in grouped.items():
        out[reason] = [
            {
                "sample_index": x.sample_index,
                "video_id": x.video_id,
                "num_frames": len(x.frame_indices_in_interval),
                "min_delta_frames": x.min_delta_frames,
                "max_delta_frames": x.max_delta_frames,
            }
            for x in items[:max_examples]
        ]
    return out