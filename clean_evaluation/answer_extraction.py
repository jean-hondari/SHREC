import re
from typing import Optional


LETTER_ANSWER_RE = re.compile(r"answer\s*:\s*([A-G](?:\s*,\s*[A-G])*)", re.IGNORECASE)
NUMBER_ANSWER_RE = re.compile(r"answer\s*:\s*\((\d)\)", re.IGNORECASE)


def extract_answer_choice(task_type: str, response: str) -> Optional[str]:
    if response is None:
        return None

    text = str(response).strip()
    if not text:
        return None

    lower_task = task_type.lower()

    if "error_vs_competence" in lower_task or "attribute" in lower_task:
        m = LETTER_ANSWER_RE.search(text)
        if m:
            vals = [x.strip().upper() for x in m.group(1).split(",")]
            return ",".join(vals)

    if "rationale" in lower_task or "correction" in lower_task:
        m = NUMBER_ANSWER_RE.search(text)
        if m:
            return f"({m.group(1)})"

    return None