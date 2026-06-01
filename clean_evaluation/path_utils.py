from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output"


def resolve_output_root(output_dir_arg: str | None) -> Path:
    if output_dir_arg is None or str(output_dir_arg).strip() == "":
        return DEFAULT_OUTPUT_ROOT

    p = Path(output_dir_arg)
    if p.is_absolute():
        return p

    # Make relative paths resolve from repository/script location, not shell cwd
    return (SCRIPT_DIR / p).resolve()


def build_structured_output_dir(output_root: Path, task_type: str, model: str) -> Path:
    task_stem = task_type.replace(".pickle", "")
    return output_root / task_stem / model