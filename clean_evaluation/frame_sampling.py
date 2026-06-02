import math
import os
from pathlib import Path
from typing import List, Optional


def _parse_frame_number(frame_path: str) -> Optional[int]:
    stem = Path(frame_path).stem
    try:
        return int(stem)
    except ValueError:
        return None


def load_interval_frame_paths_from_folder(
    images_dir: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> List[str]:
    if fps is None or fps <= 0:
        return []

    frame_dir = os.path.join(images_dir, str(video_id))
    if not os.path.isdir(frame_dir):
        return []

    start_frame = max(0, int(math.floor(float(start_sec) * float(fps))))
    end_frame = max(start_frame, int(math.ceil(float(end_sec) * float(fps))))

    candidates = []
    for name in os.listdir(frame_dir):
        if not (name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")):
            continue

        full_path = os.path.join(frame_dir, name)
        frame_num = _parse_frame_number(full_path)
        if frame_num is None:
            continue

        if start_frame <= frame_num <= end_frame:
            candidates.append((frame_num, full_path))

    candidates.sort(key=lambda x: x[0])
    return [path for _, path in candidates]


def sample_evenly_from_candidates(frame_paths: List[str], max_samples: int) -> List[str]:
    if not frame_paths:
        return []

    if max_samples <= 1:
        return [frame_paths[len(frame_paths) // 2]]

    if len(frame_paths) <= max_samples:
        return frame_paths

    sampled = []
    n = len(frame_paths)

    for i in range(max_samples):
        idx = round(i * (n - 1) / (max_samples - 1))
        sampled.append(frame_paths[idx])

    # preserve order, remove accidental duplicates
    deduped = []
    seen = set()
    for p in sampled:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    if not deduped:
        return [frame_paths[len(frame_paths) // 2]]

    return deduped


def get_sampled_interval_frames(
    images_dir: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    fps: float,
    max_samples: int,
) -> List[str]:
    interval_frames = load_interval_frame_paths_from_folder(
        images_dir=images_dir,
        video_id=video_id,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
    )

    if not interval_frames:
        return []

    sampled = sample_evenly_from_candidates(interval_frames, max_samples=max_samples)

    if len(sampled) == 0 and len(interval_frames) > 0:
        return [interval_frames[len(interval_frames) // 2]]

    return sampled