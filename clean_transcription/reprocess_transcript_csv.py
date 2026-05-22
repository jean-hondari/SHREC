#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError("Empty filename encountered.")

    path_name = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path_name).strip("._")
    if not safe:
        raise ValueError(f"Filename became empty after sanitization: {name}")
    return safe


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    print("\nRUN:", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=True,
    )


def write_raw_transcripts(
    df: pd.DataFrame,
    raw_dir: Path,
    filename_column: str,
    transcript_column: str,
) -> Dict[int, str]:
    ensure_dir(raw_dir)

    row_to_safe_name: Dict[int, str] = {}
    used_names = set()

    for idx, row in df.iterrows():
        original_name = row[filename_column]
        transcript = row[transcript_column]

        if pd.isna(transcript):
            transcript = ""

        safe_name = sanitize_filename(original_name)

        if safe_name in used_names:
            suffix = 1
            candidate = f"{safe_name}_{suffix}"
            while candidate in used_names:
                suffix += 1
                candidate = f"{safe_name}_{suffix}"
            safe_name = candidate

        used_names.add(safe_name)
        row_to_safe_name[idx] = safe_name

        output_path = raw_dir / f"{safe_name}.txt"
        output_path.write_text(str(transcript), encoding="utf-8")

    return row_to_safe_name


def run_parse(
    base_dir: Path,
    raw_dir: Path,
    parsed_dir: Path,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        str(base_dir / "parse_transcript.py"),
        str(raw_dir),
        "-o",
        str(parsed_dir),
        "--pretty",
    ]
    if overwrite:
        cmd.append("--overwrite")

    result = run_command(cmd)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def run_diarization(
    base_dir: Path,
    parsed_dir: Path,
    diarized_dir: Path,
    model: str,
    diarize_method: str,
    history_size: int,
    post_size: int,
    sleep_seconds: float,
    max_retries: int,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        str(base_dir / "speaker_diarization.py"),
        "--input",
        str(parsed_dir),
        "-o",
        str(diarized_dir),
        "--model",
        model,
        "--diarize-method",
        diarize_method,
        "--history-size",
        str(history_size),
        "--post-size",
        str(post_size),
        "--sleep",
        str(sleep_seconds),
        "--max-retries",
        str(max_retries),
    ]
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            "WARNING: speaker_diarization.py returned non-zero exit code. "
            "Continuing so successfully processed files can still be collected.",
            file=sys.stderr,
        )


def render_timestamped_transcript(
    base_dir: Path,
    diarized_json_path: Path,
) -> str:
    cmd = [
        sys.executable,
        str(base_dir / "print_diarization_change.py"),
        str(diarized_json_path),
        "--timestamped-transcript",
    ]

    result = run_command(cmd)
    output = result.stdout

    lines = output.splitlines()

    filtered_lines = []
    skipping_header = True
    for line in lines:
        if skipping_header:
            if line.startswith("FILE: "):
                continue
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            skipping_header = False

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def collect_new_transcripts(
    base_dir: Path,
    diarized_dir: Path,
    timestamped_dir: Path,
    row_to_safe_name: Dict[int, str],
    diarize_method: str,
) -> Dict[str, str]:
    ensure_dir(timestamped_dir)

    safe_name_to_transcript: Dict[str, str] = {}

    for safe_name in row_to_safe_name.values():
        diarized_json = diarized_dir / f"{safe_name}.{diarize_method}.json"
        output_txt = timestamped_dir / f"{safe_name}.txt"

        if not diarized_json.exists():
            safe_name_to_transcript[safe_name] = ""
            continue

        transcript_text = render_timestamped_transcript(base_dir, diarized_json)
        output_txt.write_text(transcript_text, encoding="utf-8")
        safe_name_to_transcript[safe_name] = transcript_text

    return safe_name_to_transcript


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprocess transcript CSV by running parse -> diarize -> timestamped transcript pipeline."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Working directory for intermediate files.",
    )
    parser.add_argument(
        "--filename-column",
        default="file_name",
        help="Column containing per-row file names.",
    )
    parser.add_argument(
        "--transcript-column",
        default="transcript",
        help="Column containing original transcript text.",
    )
    parser.add_argument(
        "--new-transcript-column",
        default="new transcript",
        help="Column name for the generated transcript.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name for diarization.",
    )
    parser.add_argument(
        "--diarize-method",
        choices=["pre_windowed_context", "pre_post_windowed_context"],
        default="pre_post_windowed_context",
        help="Diarization method.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=10,
        help="History size for diarization context.",
    )
    parser.add_argument(
        "--post-size",
        type=int,
        default=5,
        help="Post context size for pre_post_windowed_context.",
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
        help="Maximum retries per sentence during diarization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing intermediate files.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    work_dir = Path(args.work_dir).resolve()

    raw_dir = work_dir / "raw_txt"
    parsed_dir = work_dir / "parsed"
    diarized_dir = work_dir / "diarized"
    timestamped_dir = work_dir / "timestamped_txt"

    ensure_dir(work_dir)
    ensure_dir(raw_dir)
    ensure_dir(parsed_dir)
    ensure_dir(diarized_dir)
    ensure_dir(timestamped_dir)

    df = pd.read_csv(input_csv)

    if args.filename_column not in df.columns:
        raise ValueError(f"Missing required column: {args.filename_column}")

    if args.transcript_column not in df.columns:
        raise ValueError(f"Missing required column: {args.transcript_column}")

    row_to_safe_name = write_raw_transcripts(
        df=df,
        raw_dir=raw_dir,
        filename_column=args.filename_column,
        transcript_column=args.transcript_column,
    )

    run_parse(
        base_dir=base_dir,
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        overwrite=args.overwrite,
    )

    run_diarization(
        base_dir=base_dir,
        parsed_dir=parsed_dir,
        diarized_dir=diarized_dir,
        model=args.model,
        diarize_method=args.diarize_method,
        history_size=args.history_size,
        post_size=args.post_size,
        sleep_seconds=args.sleep,
        max_retries=args.max_retries,
        overwrite=args.overwrite,
    )

    safe_name_to_transcript = collect_new_transcripts(
        base_dir=base_dir,
        diarized_dir=diarized_dir,
        timestamped_dir=timestamped_dir,
        row_to_safe_name=row_to_safe_name,
        diarize_method=args.diarize_method,
    )

    df[args.new_transcript_column] = [
        safe_name_to_transcript.get(row_to_safe_name[idx], "")
        for idx in df.index
    ]

    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"SAVED OUTPUT CSV: {output_csv}")


if __name__ == "__main__":
    main()