#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


VALID_SPEAKERS = ["AI Agent", "User A"]


SYSTEM_PROMPT = """
The social robotic agent is designed to be a social positive psychology coach
that delivers interactive positive psychology interventions and provides other
useful skills to build rapport with college students.

You are given the Conversation History between a social robotic agent
(AI Agent) and a participant (User A).

Your task:
Determine the TRUE AUDIO SPEAKER of the current sentence.

Rules:
- There are only two valid speaker labels: "AI Agent" and "User A".
- Return the true audio speaker, not the person quoted inside the sentence.
- Quoted or reported speech does not mean the audio speaker changed.
- Do not rewrite transcript text.
- Do not change timestamps.
- Previous context may include already corrected speaker labels.
- Future/post context, if provided, uses original speaker labels only and may be noisy.
"""


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sentence_view(sentence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sentence_id": sentence["sentence_id"],
        "start_timestamp": sentence["start_timestamp"],
        "end_timestamp": sentence["end_timestamp"],
        "original_speaker_labels": sentence.get("speaker_labels", []),
        "modified_speaker_label": sentence.get("modified_speaker_label"),
        "modified_speaker_confidence": sentence.get("modified_speaker_confidence"),
        "text": sentence["text"],
    }


def future_sentence_view(sentence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sentence_id": sentence["sentence_id"],
        "start_timestamp": sentence["start_timestamp"],
        "end_timestamp": sentence["end_timestamp"],
        "original_speaker_labels": sentence.get("speaker_labels", []),
        "note": "This future sentence has not been corrected yet; original label may be noisy.",
        "text": sentence["text"],
    }


def call_model(
    client: OpenAI,
    model: str,
    previous_context: List[Dict[str, Any]],
    current: Dict[str, Any],
    post_context: List[Dict[str, Any]],
    diarize_method: str,
) -> Dict[str, Any]:
    payload = {
        "diarize_method": diarize_method,
        "valid_speaker_labels": VALID_SPEAKERS,
        "previous_context": previous_context,
        "current_sentence": current,
        "post_context": post_context,
        "post_context_warning": (
            "Post context uses original speaker labels only and may be noisy. "
            "Use it for conversational continuity, but do not fully trust its labels."
        ),
    }

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "speaker_label_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "speaker_label": {
                            "type": "string",
                            "enum": VALID_SPEAKERS,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["speaker_label", "confidence"],
                },
            }
        },
    )

    return json.loads(response.output_text)


def already_processed(data: Dict[str, Any]) -> bool:
    sentences = data.get("sentences", [])
    if not sentences:
        return False

    return all(
        "modified_speaker_label" in sentence
        and "modified_speaker_confidence" in sentence
        for sentence in sentences
    )


def is_sentence_processed(sentence: Dict[str, Any]) -> bool:
    return (
        "modified_speaker_label" in sentence
        and "modified_speaker_confidence" in sentence
    )


def build_processed_history(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []

    for sentence in sentences:
        if is_sentence_processed(sentence):
            history.append(sentence_view(sentence))
        else:
            break

    return history


def init_metadata(
    data: Dict[str, Any],
    *,
    model: str,
    history_size: int,
    post_size: int,
    diarize_method: str,
) -> None:
    data.setdefault("metadata", {})
    metadata = data["metadata"]

    metadata["speaker_relabel_model"] = model
    metadata["speaker_relabel_history_size"] = history_size
    metadata["speaker_relabel_post_size"] = (
        post_size if diarize_method == "pre_post_windowed_context" else 0
    )
    metadata["speaker_relabel_diarize_method"] = diarize_method
    metadata["valid_modified_speaker_labels"] = VALID_SPEAKERS
    metadata.setdefault("speaker_relabel_status", "in_progress")
    metadata.setdefault("speaker_relabel_error_count", 0)
    metadata.setdefault("speaker_relabel_last_error", None)
    metadata.setdefault("speaker_relabel_last_error_type", None)
    metadata.setdefault("speaker_relabel_last_processed_sentence_id", None)
    metadata.setdefault("speaker_relabel_completed_sentences", 0)
    metadata.setdefault("speaker_relabel_total_sentences", len(data.get("sentences", [])))


def update_progress_metadata(data: Dict[str, Any], sentence_id: Optional[int]) -> None:
    metadata = data.setdefault("metadata", {})
    sentences = data.get("sentences", [])

    completed = sum(1 for sentence in sentences if is_sentence_processed(sentence))
    metadata["speaker_relabel_completed_sentences"] = completed
    metadata["speaker_relabel_total_sentences"] = len(sentences)
    metadata["speaker_relabel_last_processed_sentence_id"] = sentence_id


def classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()

    if "rate limit" in text or "429" in text or "quota" in text:
        return "rate_limit"

    if "timeout" in text or "timed out" in text:
        return "timeout"

    if "connection" in text or "api connection" in text or "connect" in text:
        return "connection"

    if "server" in text or "500" in text or "502" in text or "503" in text or "504" in text:
        return "server"

    if "json" in text or "decode" in text or "schema" in text:
        return "response_format"

    return "unknown"


def compute_backoff_seconds(error_type: str, attempt_index: int) -> float:
    if error_type == "rate_limit":
        base = 10
        cap = 60
    elif error_type in {"timeout", "connection", "server"}:
        base = 10
        cap = 30
    elif error_type == "response_format":
        base = 5
        cap = 15
    else:
        base = 5
        cap = 20

    delay = min(cap, base * (2 ** max(0, attempt_index - 1)))
    jitter = random.uniform(0, min(3, delay / 4))
    return delay + jitter


def safe_error_message(exc: Exception, max_len: int = 500) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if len(message) > max_len:
        return message[: max_len - 3] + "..."
    return message


def relabel_sentence_with_retry(
    client: OpenAI,
    model: str,
    previous_context: List[Dict[str, Any]],
    current: Dict[str, Any],
    post_context: List[Dict[str, Any]],
    diarize_method: str,
    max_retries: int,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_model(
                client=client,
                model=model,
                previous_context=previous_context,
                current=current,
                post_context=post_context,
                diarize_method=diarize_method,
            )
        except Exception as exc:
            last_exc = exc
            error_type = classify_exception(exc)

            if attempt >= max_retries:
                raise

            sleep_seconds = compute_backoff_seconds(error_type, attempt)
            print(
                f"WARNING retrying sentence_id={current.get('sentence_id')} "
                f"attempt={attempt}/{max_retries} "
                f"error_type={error_type} "
                f"sleep={sleep_seconds:.1f}s "
                f"error={safe_error_message(exc)}"
            )
            time.sleep(sleep_seconds)

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Unexpected retry flow reached without result or exception.")


def relabel_file(
    input_path: Path,
    output_path: Path,
    model: str,
    history_size: int,
    post_size: int,
    sleep_seconds: float,
    overwrite: bool,
    diarize_method: str,
    max_retries: int,
) -> None:
    source_data = load_json(input_path)

    if output_path.exists() and not overwrite:
        data = load_json(output_path)
        print(f"RESUME from existing output: {output_path}")
    else:
        data = source_data

    init_metadata(
        data,
        model=model,
        history_size=history_size,
        post_size=post_size,
        diarize_method=diarize_method,
    )

    sentences = data.get("sentences", [])
    if not sentences:
        data["metadata"]["speaker_relabel_status"] = "completed"
        save_json(output_path, data)
        print(f"SAVED empty transcript: {output_path}")
        return

    if output_path.exists() and not overwrite and already_processed(data):
        data["metadata"]["speaker_relabel_status"] = "completed"
        update_progress_metadata(
            data,
            sentence_id=sentences[-1].get("sentence_id") if sentences else None,
        )
        save_json(output_path, data)
        print(f"SKIP already processed: {output_path}")
        return

    client = OpenAI()
    processed_history = build_processed_history(sentences)

    start_index = len(processed_history)
    if start_index > 0:
        print(
            f"Found partial progress for {input_path.name}: "
            f"{start_index}/{len(sentences)} sentences already processed"
        )

    for i in range(start_index, len(sentences)):
        sentence = sentences[i]

        previous_context = processed_history[-history_size:]
        current = sentence_view(sentence)

        if diarize_method == "pre_post_windowed_context":
            post_context = [
                future_sentence_view(s)
                for s in sentences[i + 1 : i + 1 + post_size]
            ]
        else:
            post_context = []

        try:
            result = relabel_sentence_with_retry(
                client=client,
                model=model,
                previous_context=previous_context,
                current=current,
                post_context=post_context,
                diarize_method=diarize_method,
                max_retries=max_retries,
            )
        except Exception as exc:
            metadata = data.setdefault("metadata", {})
            metadata["speaker_relabel_status"] = "failed"
            metadata["speaker_relabel_error_count"] = metadata.get("speaker_relabel_error_count", 0) + 1
            metadata["speaker_relabel_last_error"] = safe_error_message(exc)
            metadata["speaker_relabel_last_error_type"] = classify_exception(exc)
            update_progress_metadata(
                data,
                sentence_id=sentences[i - 1].get("sentence_id") if i > 0 else None,
            )
            save_json(output_path, data)
            print(
                f"ERROR failed on {input_path.name} "
                f"sentence_id={sentence.get('sentence_id')} "
                f"error_type={metadata['speaker_relabel_last_error_type']} "
                f"error={metadata['speaker_relabel_last_error']}"
            )
            raise

        sentence["modified_speaker_label"] = result["speaker_label"]
        sentence["modified_speaker_confidence"] = result["confidence"]

        processed_history.append(sentence_view(sentence))

        metadata = data.setdefault("metadata", {})
        metadata["speaker_relabel_status"] = "in_progress"
        metadata["speaker_relabel_last_error"] = None
        metadata["speaker_relabel_last_error_type"] = None
        update_progress_metadata(data, sentence_id=sentence.get("sentence_id"))

        save_json(output_path, data)

        print(
            f"{input_path.name} [{i + 1}/{len(sentences)}] "
            f"sentence_id={sentence['sentence_id']} -> "
            f"{result['speaker_label']} ({result['confidence']:.2f})"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    data["metadata"]["speaker_relabel_status"] = "completed"
    update_progress_metadata(
        data,
        sentence_id=sentences[-1].get("sentence_id") if sentences else None,
    )
    save_json(output_path, data)
    print(f"SAVED: {output_path}")


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel transcript sentence speaker labels using OpenAI API."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory where modified JSON files will be saved.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--diarize-method",
        choices=[
            "pre_windowed_context",
            "pre_post_windowed_context",
        ],
        default="pre_windowed_context",
        help="Diarization context method.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=10,
        help="Number of previous sentences to include as context.",
    )
    parser.add_argument(
        "--post-size",
        type=int,
        default=5,
        help="Number of future sentences to include for pre_post_windowed_context.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between successful API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per sentence when API or transient errors occur.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if output already exists.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = collect_input_files(input_path)

    had_failure = False

    for file_path in input_files:
        method_suffix = args.diarize_method
        output_filename = f"{file_path.stem}.{method_suffix}.json"
        output_path = output_dir / output_filename

        try:
            relabel_file(
                input_path=file_path,
                output_path=output_path,
                model=args.model,
                history_size=args.history_size,
                post_size=args.post_size,
                sleep_seconds=args.sleep,
                overwrite=args.overwrite,
                diarize_method=args.diarize_method,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            had_failure = True
            print(
                f"FILE FAILED: {file_path.name} | "
                f"{type(exc).__name__}: {exc}"
            )

    if had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()