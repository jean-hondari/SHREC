from typing import List


def build_task_prefix(task_type: str) -> str:
    lower = task_type.lower()

    if "dorm" in lower or "home" in lower:
        return (
            "The social robotic agent is Jibo, a positive psychology coach. "
            "Jibo delivers interactive positive psychology activities and uses conversation "
            "to build rapport with users.\n\n"
            "You are given the conversation history between Jibo and a participant. "
            "If images are provided, use them together with the conversation history."
        )

    return (
        "The social robotic agent is Jibo, a social support companion. "
        "Jibo facilitates emotionally relevant storytelling and provides supportive responses "
        "to help users feel connection and belonging.\n\n"
        "You are given the conversation history between Jibo and a participant. "
        "If images are provided, use them together with the conversation history."
    )


def infer_social_label(sample: dict) -> str:
    error_value = sample.get("error")
    if error_value is True:
        return "Social Error"
    if error_value is False:
        return "Social Competence"
    return "the labeled social behavior"


def build_error_vs_competence_prompt(task_type: str, transcript: str) -> str:
    prefix = build_task_prefix(task_type)

    return (
        f"{prefix}\n\n"
        "Task:\n"
        "Determine whether Jibo's behavior in the interaction is an instance of Social Competence or Social Error.\n\n"
        "Definitions:\n"
        "Social Competence means Jibo successfully conducts the social interaction. "
        "This includes noticing relevant social or emotional cues, interpreting them appropriately, "
        "and responding in a socially expected or helpful way.\n\n"
        "Social Error means Jibo behaves in a way that violates social expectations, misunderstands "
        "the user or situation, responds inappropriately, or degrades the user's perception of "
        "Jibo's social competence.\n\n"
        "Choose exactly one answer:\n"
        "(A) Social Competence\n"
        "(B) Social Error\n\n"
        "Conversation History:\n"
        f"{transcript}\n\n"
        "Answer format:\n"
        "Answer: A"
    )


def build_attribute_prompt(task_type: str, transcript: str, sample: dict) -> str:
    prefix = build_task_prefix(task_type)
    social_label = infer_social_label(sample)

    return (
        f"{prefix}\n\n"
        "Task:\n"
        f"The following interaction has already been labeled as an instance of {social_label}.\n\n"
        "Identify which social attribute or attributes are most relevant to Jibo's behavior in this interaction. "
        "Multiple attributes may apply.\n\n"
        "Social attributes:\n"
        "(A) Emotions: recognizing, interpreting, or responding to the user's emotional state.\n"
        "(B) Engagement: recognizing whether the user is interested, attentive, participating, disengaged, or leaving the interaction.\n"
        "(C) Conversational Mechanics: managing turn-taking, timing, interruptions, silence, response delay, topic flow, or conversation repair.\n"
        "(D) Knowledge State: tracking what Jibo or the user knows, remembers, believes, has already said, or still needs to know.\n"
        "(E) User Intention: understanding the user's goal, request, instruction, readiness, refusal, or intended meaning.\n"
        "(F) Social Context and Relationships: understanding roles, relationship dynamics, rapport, support, trust, or the social purpose of the interaction.\n"
        "(G) Social Norms and Routines: following expected politeness, apologies, greetings, closings, norms, rituals, or socially appropriate behavior.\n\n"
        "Choose all applicable answers from A, B, C, D, E, F, G.\n\n"
        "Conversation History:\n"
        f"{transcript}\n\n"
        "Answer format:\n"
        "Answer: A,C,F"
    )


def _build_choice_block(choices: List[str], heading: str) -> str:
    lines = [f"{heading}:"]
    for idx, choice_text in enumerate(choices, start=1):
        lines.append(f"({idx}) {choice_text}")
    return "\n".join(lines)


def build_rationale_error_prompt(task_type: str, transcript: str, choices: List[str]) -> str:
    prefix = build_task_prefix(task_type)
    reasons_block = _build_choice_block(choices, "Reasons")

    return (
        f"{prefix}\n\n"
        "Task:\n"
        "The following interaction has already been labeled as a Social Error.\n\n"
        "Select the reason that best explains why Jibo's behavior is a Social Error.\n\n"
        "Conversation History:\n"
        f"{transcript}\n\n"
        f"{reasons_block}\n\n"
        "Choose exactly one answer.\n\n"
        "Answer format:\n"
        "Answer: (1)"
    )


def build_rationale_competence_prompt(task_type: str, transcript: str, choices: List[str]) -> str:
    prefix = build_task_prefix(task_type)
    reasons_block = _build_choice_block(choices, "Reasons")

    return (
        f"{prefix}\n\n"
        "Task:\n"
        "The following interaction has already been labeled as Social Competence.\n\n"
        "Select the reason that best explains why Jibo's behavior is Social Competence.\n\n"
        "Conversation History:\n"
        f"{transcript}\n\n"
        f"{reasons_block}\n\n"
        "Choose exactly one answer.\n\n"
        "Answer format:\n"
        "Answer: (1)"
    )


def build_correction_prompt(task_type: str, transcript: str, choices: List[str]) -> str:
    prefix = build_task_prefix(task_type)
    behaviors_block = _build_choice_block(choices, "Behaviors")

    return (
        f"{prefix}\n\n"
        "Task:\n"
        "The following interaction has already been labeled as a Social Error.\n\n"
        "Select which behavior Jibo should have done instead.\n\n"
        "Conversation History:\n"
        f"{transcript}\n\n"
        f"{behaviors_block}\n\n"
        "Choose exactly one answer.\n\n"
        "Answer format:\n"
        "Answer: (1)"
    )