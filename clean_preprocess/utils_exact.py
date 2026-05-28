
import pdb
import ast
import pandas as pd
import os
import cv2
import base64
from natsort import natsorted
def return_exact_convos(start_time, end_time, df, turn=False, more_context=True, buffer_seconds=0.0):
    start_time = max(0.0, float(start_time) - float(buffer_seconds))
    end_time = float(end_time) + float(buffer_seconds)

    if df.empty:
        return ""

    agent_id = 'AI Agent:'

    if turn:
        if (df['timestamp'] < start_time)[df['timestamp'] < start_time].empty:
            start_index = 0
        else:
            start_index = (df['timestamp'] < start_time)[df['timestamp'] < start_time].index[-1]

        if (~(df['timestamp'] > end_time))[(~(df['timestamp'] > end_time))].empty:
            end_index = len(df) - 1
        else:
            end_index = (~(df['timestamp'] > end_time))[(~(df['timestamp'] > end_time))].index[-1]

        if df['speaker'].iloc[start_index] == agent_id:
            if start_index == 0:
                curr_df = df.iloc[start_index:start_index + 1, :]
            else:
                curr_df = df.iloc[start_index - 1:start_index + 1, :]
        else:
            if start_index == len(df) - 1:
                curr_df = df.iloc[start_index:start_index + 1, :]
            else:
                curr_df = df.iloc[start_index:start_index + 2, :]

        conversation = "\n".join(f"{row['speaker']} {row['text']}" for _, row in curr_df.iterrows())
        return conversation

    # exact mode: keep only rows whose timestamps are inside the buffered window
    curr_df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)].copy()

    if curr_df.empty:
        return ""

    # Reconstruct text grouped by contiguous speaker segments
    conversation_lines = []
    current_speaker = None
    current_tokens = []

    for _, row in curr_df.iterrows():
        speaker = row["speaker"]
        text = str(row["text"]).strip()

        if text == "":
            continue

        if current_speaker is None:
            current_speaker = speaker
            current_tokens = [text]
        elif speaker == current_speaker:
            current_tokens.append(text)
        else:
            conversation_lines.append(f"{current_speaker} {' '.join(current_tokens).strip()}")
            current_speaker = speaker
            current_tokens = [text]

    if current_speaker is not None and current_tokens:
        conversation_lines.append(f"{current_speaker} {' '.join(current_tokens).strip()}")

    return "\n".join(conversation_lines)
import re
import pandas as pd

TIMESTAMP_WORD_RE = re.compile(r"\((\d{2}:\d{2}:\d{2})\)\s*([^\s()]+)")
SPEAKER_RE = re.compile(r"^\s*([^:\n]+):\s*$")
END_PUNCT = {".", "?", "!"}


def timestamp_str_to_seconds(ts):
    hh, mm, ss = ts.split(":")
    return float(hh) * 3600 + float(mm) * 60 + float(ss)


def normalize_speaker_label(label):
    label = str(label).strip()
    if label == "User A":
        return "User"
    if label == "AI Agent":
        return "Agent"
    return label


def parse_cleaned_transcript_words(text):
    """
    Parse transcript exactly in the format produced by clean_transcription tools.

    Expected structure:
        User A:
        (00:00:12) hello (00:00:13) there.

        AI Agent:
        (00:00:15) hi (00:00:16) there.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    words = []
    current_speaker = None
    word_id = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        speaker_match = SPEAKER_RE.match(stripped)
        if speaker_match and "(" not in stripped:
            current_speaker = speaker_match.group(1).strip()
            continue

        if current_speaker is None:
            continue

        matches = TIMESTAMP_WORD_RE.findall(stripped)
        if not matches:
            continue

        for ts, word in matches:
            clean_word = word.strip()
            if clean_word == "":
                continue

            words.append(
                {
                    "word_id": word_id,
                    "word": clean_word,
                    "timestamp": ts,
                    "timestamp_seconds": timestamp_str_to_seconds(ts),
                    "speaker_label": current_speaker,
                    "line_no": line_no,
                }
            )
            word_id += 1

    return words


def build_sentences_from_cleaned_words(words):
    """
    Same sentence segmentation rule as clean_transcription/parse_transcript.py:
    - sentence ends at punctuation
    - or at speaker boundary
    """
    if not words:
        return []

    sentences = []
    current_words = []

    for i, word_obj in enumerate(words):
        current_words.append(word_obj)

        word = word_obj["word"]
        ends_sentence = word.endswith(tuple(END_PUNCT))

        next_word = words[i + 1] if i + 1 < len(words) else None
        speaker_changes_next = (
            next_word is not None
            and next_word["speaker_label"] != word_obj["speaker_label"]
        )

        if ends_sentence or speaker_changes_next:
            sentences.append(
                {
                    "speaker": normalize_speaker_label(current_words[0]["speaker_label"]),
                    "start_timestamp": current_words[0]["timestamp_seconds"],
                    "end_timestamp": current_words[-1]["timestamp_seconds"],
                    "text": " ".join(w["word"] for w in current_words).strip(),
                    "words": current_words[:],
                }
            )
            current_words = []

    if current_words:
        sentences.append(
            {
                "speaker": normalize_speaker_label(current_words[0]["speaker_label"]),
                "start_timestamp": current_words[0]["timestamp_seconds"],
                "end_timestamp": current_words[-1]["timestamp_seconds"],
                "text": " ".join(w["word"] for w in current_words).strip(),
                "words": current_words[:],
            }
        )

    return sentences


def format_float_seconds(value):
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def format_sentence_timestamp_transcript(sentences):
    """
    Output:
        Speaker
        start-end sentence

        Speaker
        start-end sentence
    """
    if not sentences:
        return ""

    blocks = []
    current_speaker = None
    current_lines = []

    for sent in sentences:
        line = (
                f"{format_mmss(sent['start_timestamp'])}-"
                f"{format_mmss(sent['end_timestamp'])} "
                f"{sent['text']}"
            )

        if current_speaker is None:
            current_speaker = sent["speaker"]
            current_lines = [current_speaker, line]
        elif sent["speaker"] == current_speaker:
            current_lines.append(line)
        else:
            blocks.append("\n".join(current_lines))
            current_speaker = sent["speaker"]
            current_lines = [current_speaker, line]

    if current_lines:
        blocks.append("\n".join(current_lines))

    return "\n\n".join(blocks)


def get_transcript(timestamp_dict, transcriptions, session_name=None, data_path=None, time_type=None,
                   mode="turn", buffer_seconds=0.0):
    """
    Extract transcript using cleaned transcript format.

    Important:
    - buffer_seconds is only for transcript retrieval
    - sentence timestamps come from cleaned transcript word timestamps
    - no made-up Unknown speaker labels
    """
    start_time = timestamp_dict["start"]
    end_time = timestamp_dict["end"]

    if start_time is None or end_time is None:
        return ""

    if not isinstance(transcriptions, str) or transcriptions.strip() == "":
        return ""

    buffered_start = float(start_time) - float(buffer_seconds)
    buffered_end = float(end_time) + float(buffer_seconds)

    words = parse_cleaned_transcript_words(transcriptions)
    if not words:
        return ""

    sentences = build_sentences_from_cleaned_words(words)
    if not sentences:
        return ""

    selected_sentences = []
    for sent in sentences:
        sent_start = sent["start_timestamp"]
        sent_end = sent["end_timestamp"]

        overlaps = not (sent_end < buffered_start or sent_start > buffered_end)
        if overlaps:
            selected_sentences.append(sent)

    if not selected_sentences:
        return ""

    if mode == "exact":
        return format_sentence_timestamp_transcript(selected_sentences)

    if mode == "turn":
        # Conservative turn approximation:
        # include one sentence before/after around the first selected sentence.
        first_idx = None
        for i, sent in enumerate(sentences):
            if sent in selected_sentences:
                first_idx = i
                break

        if first_idx is None:
            return ""

        left = max(0, first_idx - 1)
        right = min(len(sentences), first_idx + 2)
        return format_sentence_timestamp_transcript(sentences[left:right])

    raise ValueError(f"Unsupported transcript mode: {mode}")
def format_mmss(value):
    """
    Format seconds as mm:ss.
    """
    total_seconds = int(float(value))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def format_sentence_timestamp_transcript(sentences):
    """
    Convert selected sentence entries into the requested output format:

        Speaker
        mm:ss-mm:ss sentence 1
        mm:ss-mm:ss sentence 2

        Speaker
        ...

    Consecutive sentences from the same speaker are grouped into one block.
    """
    if not sentences:
        return ""

    grouped_blocks = []
    current_speaker = None
    current_lines = []

    for sent in sentences:
        speaker = sent["speaker"]
        line = (
            f"{format_mmss(sent['start_timestamp'])}-"
            f"{format_mmss(sent['end_timestamp'])} "
            f"{sent['text']}"
        )

        if current_speaker is None:
            current_speaker = speaker
            current_lines = [speaker, line]
        elif speaker == current_speaker:
            current_lines.append(line)
        else:
            grouped_blocks.append("\n".join(current_lines))
            current_speaker = speaker
            current_lines = [speaker, line]

    if current_lines:
        grouped_blocks.append("\n".join(current_lines))

    return "\n\n".join(grouped_blocks)
def _valid_timestamp(interval):
    ts = interval.get("timestamp", {})
    start = ts.get("start")
    end = ts.get("end")
    return not isinstance(start, list) and not isinstance(end, list) and start is not None and end is not None

def find_overlapping_interval_groups_pair(intervals1, intervals2):
    intervals1 = [i for i in intervals1 if _valid_timestamp(i)]
    intervals2 = [i for i in intervals2 if _valid_timestamp(i)]

    sorted_intervals1 = sorted(
        enumerate(intervals1),
        key=lambda x: x[1]["timestamp"]["start"]
    )
    sorted_intervals2 = sorted(
        enumerate(intervals2),
        key=lambda x: x[1]["timestamp"]["start"]
    )

    overlapping_groups = []

    for current_index, current_interval in sorted_intervals1:

        # if current_interval['timestamp']['start'] != None and current_interval['timestamp']['end'] != None :
        #     continue

        current_group = [current_interval]
        added = False 
        
        need_rerank_group = []
        for index, interval in sorted_intervals2:
            # if interval['timestamp']['start'] == None or interval['timestamp']['end']:
            #     continue
            if interval['timestamp']['start'] < current_interval['timestamp']['start'] and interval['timestamp']['end'] > current_interval['timestamp']['end'] and abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 10:
                need_rerank_group.append((index,interval))
            elif interval['timestamp']['start'] > current_interval['timestamp']['start'] and interval['timestamp']['end'] < current_interval['timestamp']['end'] and abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 10:
                need_rerank_group.append((index,interval))
            elif abs(interval['timestamp']['start'] - current_interval['timestamp']['start']) < 3:
                need_rerank_group.append((index,interval))
            elif abs(interval['timestamp']['end'] - current_interval['timestamp']['end']) < 3:
                need_rerank_group.append((index,interval))
        
        #finished populating potential matches rank by closest to end and start
        for index, match_interval in need_rerank_group:
            if current_interval['error'] == match_interval['error']:
                current_group.append(match_interval)
                sorted_intervals2.remove((index, match_interval))
                overlapping_groups.append(current_group)
                added = True 
                break
        
        if len(need_rerank_group) > 0 and not added:
            current_group.append(match_interval)
            sorted_intervals2.remove((index, match_interval))
            overlapping_groups.append(current_group)

        if len(need_rerank_group) == 0 and not added:
            overlapping_groups.append(current_group)

        
    #add remaining from interval2
    for index, interval in sorted_intervals2:
        overlapping_groups.append([interval])
    

    return overlapping_groups

def merge_intervals(intervals):
    merged_intervals = []
    for interval in sorted(intervals, key=lambda x: x['start']):
        if not merged_intervals or interval['start'] > merged_intervals[-1]['end']:
            merged_intervals.append(interval)
        else:
            merged_intervals[-1]['end'] = max(merged_intervals[-1]['end'], interval['end'])
    return merged_intervals

def find_uncovered_float_intervals(intervals, total_time_range):
    # Merge overlapping intervals
    merged_intervals = merge_intervals(intervals)
    
    # Initialize the uncovered intervals list
    uncovered_intervals = []
    
    # Initialize variables for tracking current covered range
    current_start = total_time_range['start']
    current_end = total_time_range['start']

    for interval in merged_intervals:
        if interval['start'] > current_end:
            # Gaps exist between current_end and next interval's start
            uncovered_intervals.append({'start': current_end, 'end': interval['start']})
        
        # Update current_start and current_end
        current_start = interval['start']
        current_end = max(current_end, interval['end'])
    
    # Check for uncovered time at the end of the total time range
    if current_end < total_time_range['end']:
        uncovered_intervals.append({'start': current_end, 'end': total_time_range['end']})
    
    return uncovered_intervals


def flatten(xss):
    return [x for xs in xss for x in xs]



def filter_dicts_without_subdict_tier2(dicts_list, sub_dict):
    all_other_dict = []
    for other_dict in dicts_list:
        diff = True 
        for key in sub_dict:
            if sub_dict.get(key) is True:  # Check if the value in dict1 is True
                if other_dict['attribute'].get(key) is True:  # Check if the value in dict2 is also True
                    diff = False
                    # print(f"Both dictionaries have True for key '{key}'")
                else:
                    pass
                    # print(f"dict1 has True for key '{key}', but dict2 does not")
        if diff == True:
            all_other_dict.append(other_dict)
    return all_other_dict


def filter_dicts_without_subdict(dicts_list, sub_dict):
    filtered_dicts = [d for d in dicts_list if not all(item in d.items() for item in sub_dict.items())]
    return filtered_dicts

def filter_dicts_with_subdict(dicts_list, sub_dict):
    filtered_dicts = [d for d in dicts_list if all(item in d.items() for item in sub_dict.items())]
    return filtered_dicts

def filter_dicts_with_any_true_in_subdict(list_of_dicts, match_dict):
    filtered_list = [d for d in list_of_dicts if not any(d.get(key) and value for key, value in match_dict.items())]
    return filtered_list

def remove_redundant_strings(strings_list):
    unique_strings = list(set(strings_list))
    return unique_strings


def remove_redundant_strings_id_timestamp(strings_list,other_id_list, other_timestamp_list):
    unique_strings = []
    unique_id = [] 
    unique_timestamps = []

    for i in range(len(strings_list)):
        if strings_list[i] not in unique_strings:
            unique_strings.append(strings_list[i])
            unique_id.append(other_id_list[i])
            unique_timestamps.append(other_timestamp_list[i])

    return unique_strings, unique_id, unique_timestamps



def and_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] and dict2[key]
    return result_dict

def or_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] or dict2[key]
    return result_dict

def disagree_operation(dict1, dict2):
    result_dict = {}
    for key in dict1.keys() & dict2.keys():
        result_dict[key] = dict1[key] and dict2[key]
    return result_dict


def find_all_indexes(string, substring):
    return [index for index, _ in enumerate(string) if string[index:index + len(substring)] == substring]


def get_majority(items):
    item_counts = Counter(items)
    max_frequency = max(item_counts.values())

    candidates = [item for item, count in item_counts.items() if count == max_frequency]

    if candidates:
        return random.choice(candidates)
    else:
        return None

def sample_images_from_video(video_path, timestamp_range, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    frame_step = max(1, (end_frame - start_frame) // num_images)
    current_frame = start_frame

    img_path_list = []
    for _ in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        tmp_video_path = video_path.replace("/", "_").replace(".mp4", "")
        image_path = os.path.join(output_dir, f"frame_{current_frame//fps}_{tmp_video_path}.jpg")
        cv2.imwrite(image_path, frame)
        current_frame += frame_step
        img_path_list.append(image_path)

    if len(img_path_list) == 0:
        pdb.set_trace()
        print('VIDEO CORRUPT')

    cap.release()

    return img_path_list

def process_video(video_path, timestamp_range, num_images, output_dir):
    print("video_path:", video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base64Frames = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    frame_step = max(1, (end_frame - start_frame) // num_images)
    current_frame = start_frame

    for _ in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += frame_step
    cap.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def process_video(img_paths, timestamp_range, num_images, output_dir):
    print("video_path:", video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    base64Frames = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    frame_step = max(1, (end_frame - start_frame) // num_images)
    current_frame = start_frame

    for _ in range(num_images):
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        current_frame += frame_step
    cap.release()

    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

def process_images_base64(img_paths, output_dir=None):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    base64Frames = []

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image at {img_path}")
            continue
        
        _, buffer = cv2.imencode(".jpg", img)
        base64_frame = base64.b64encode(buffer).decode("utf-8")
        base64Frames.append(base64_frame)

    print(f"Processed {len(base64Frames)} images")
    return base64Frames


def sample_images_from_video_in_sec(video_path, timestamp_range, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    start_time = timestamp_range[0]
    end_time = timestamp_range[1]
        
    step = (end_time - start_time)//num_images
    current_time = start_time

    img_path_list = []
    for _ in range(num_images):

        # Set the video position to the desired time
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

        # Read the frame at the specified time
        ret, frame = cap.read()

        if ret:
            # Generate a default filename if not provided
            tmp_video_path = video_path.replace("/", "_").replace(".mp4", "")
            image_path = os.path.join(output_dir, f"time_{current_time}_{tmp_video_path}.jpg")
            cv2.imwrite(image_path, frame)
            current_time += step
            img_path_list.append(image_path)
            print(f"Frame saved as {image_path}")

    pdb.set_trace()
    if len(img_path_list) == 0:
        print('VIDEO CORRUPT')

    cap.release()

    return img_path_list

def trim_video(video_path, timestamp_range, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_time, end_time = timestamp_range
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_video_path = os.path.join(output_dir, 'trimmed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    for current_frame in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    return output_video_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def return_None(a,b,c,d): return []

def get_frame_paths(frame_dir, start, end):
    all_files = os.listdir(frame_dir)
    frame_paths = []

    for fname in all_files:
        if fname.endswith('.png'):
            try:
                frame_idx = int(os.path.splitext(fname)[0])
                if start <= frame_idx <= end:
                    frame_paths.append(os.path.join(frame_dir, fname))
            except ValueError:
                continue  # Skip files that aren't integer-named

    frame_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_paths = natsorted(frame_paths)
    return frame_paths


import cv2
import os

def fill_and_write_video(frame_dir, output_dir, fps=24):
    os.makedirs(output_dir, exist_ok=True)

    # Get all .png files with integer-named stems
    frame_files = [
        f for f in os.listdir(frame_dir)
        if f.endswith('.png') and f[:-4].isdigit()
    ]
    
    if not frame_files:
        raise ValueError("No valid .png frames found in the directory.")

    # Map from index to full path, using actual file names
    frame_map = {
        int(f[:-4]): os.path.join(frame_dir, f)
        for f in frame_files
    }

    sorted_indices = sorted(frame_map.keys())

    # Load the first valid frame to get size
    first_frame = cv2.imread(frame_map[sorted_indices[0]])
    if first_frame is None:
        raise ValueError(f"Failed to read {frame_map[sorted_indices[0]]}")
    height, width, _ = first_frame.shape

    output_video_path = os.path.join(output_dir, 'trimmed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write repeated frames for gaps
    for i, curr_idx in enumerate(sorted_indices):
        next_idx = sorted_indices[i + 1] if i + 1 < len(sorted_indices) else curr_idx + 1
        repeat_count = next_idx - curr_idx

        frame = cv2.imread(frame_map[curr_idx])
        if frame is None:
            print(f"Warning: Could not read {frame_map[curr_idx]}, skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        for _ in range(repeat_count):
            out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")
    return output_video_path
