#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

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


def relabel_file(
    input_path: Path,
    output_path: Path,
    model: str,
    history_size: int,
    post_size: int,
    sleep_seconds: float,
    overwrite: bool,
    diarize_method: str,
) -> None:
    data = load_json(input_path)

    if output_path.exists() and not overwrite:
        existing = load_json(output_path)
        if already_processed(existing):
            print(f"SKIP already processed: {output_path}")
            return

    client = OpenAI()
    sentences = data["sentences"]
    processed_history: List[Dict[str, Any]] = []

    for i, sentence in enumerate(sentences):
        if (
            "modified_speaker_label" in sentence
            and "modified_speaker_confidence" in sentence
            and not overwrite
        ):
            processed_history.append(sentence_view(sentence))
            continue

        previous_context = processed_history[-history_size:]
        current = sentence_view(sentence)

        if diarize_method == "pre_post_windowed_context":
            post_context = [
                future_sentence_view(s)
                for s in sentences[i + 1 : i + 1 + post_size]
            ]
        else:
            post_context = []

        result = call_model(
            client=client,
            model=model,
            previous_context=previous_context,
            current=current,
            post_context=post_context,
            diarize_method=diarize_method,
        )

        sentence["modified_speaker_label"] = result["speaker_label"]
        sentence["modified_speaker_confidence"] = result["confidence"]

        processed_history.append(sentence_view(sentence))

        print(
            f"{input_path.name} [{i + 1}/{len(sentences)}] "
            f"sentence_id={sentence['sentence_id']} -> "
            f"{result['speaker_label']} ({result['confidence']:.2f})"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    data.setdefault("metadata", {})
    data["metadata"]["speaker_relabel_model"] = model
    data["metadata"]["speaker_relabel_history_size"] = history_size
    data["metadata"]["speaker_relabel_post_size"] = (
        post_size if diarize_method == "pre_post_windowed_context" else 0
    )
    data["metadata"]["speaker_relabel_diarize_method"] = diarize_method
    data["metadata"]["valid_modified_speaker_labels"] = VALID_SPEAKERS

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
        help="Seconds to sleep between API calls.",
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

    for file_path in input_files:
        method_suffix = args.diarize_method

        output_filename = (
            f"{file_path.stem}.{method_suffix}.json"
        )

        output_path = output_dir / output_filename

        relabel_file(
            input_path=file_path,
            output_path=output_path,
            model=args.model,
            history_size=args.history_size,
            post_size=args.post_size,
            sleep_seconds=args.sleep,
            overwrite=args.overwrite,
            diarize_method=args.diarize_method,
        )


if __name__ == "__main__":
    main()