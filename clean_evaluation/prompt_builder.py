from typing import List


def build_task_prefix(task_type: str) -> str:
    lower = task_type.lower()

    if "dorm" in lower or "home" in lower:
        return (
            "The social robotic agent is designed to be a social positive psychology coach "
            "that delivers interactive positive psychology interventions and provide other "
            "useful skills to build rapport with college students. "
        )

    return (
        "The social robotic agent is designed to be a social support companion that "
        "facilitates the exchange of emotionally relevant stories and employs narrative "
        "therapy techniques to enhance feelings of connection and belonging. "
    )


def build_inputs_label(use_images: bool) -> str:
    return "Images and Conversation History" if use_images else "Conversation History"

def infer_social_label(sample: dict) -> str:
    error_value = sample.get("error")
    if error_value is True:
        return "Social Error"
    if error_value is False:
        return "Social Competence"
    return "the labeled social behavior"


def _build_choice_block(choices: List[str], heading: str) -> str:
    lines = [f"{heading}:"]
    for idx, choice_text in enumerate(choices, start=1):
        lines.append(f"({idx}) {choice_text}")
    return "\n".join(lines)


def build_error_vs_competence_prompt(task_type: str, transcript: str, use_images: bool) -> str:
    prefix = build_task_prefix(task_type)
    inputs = build_inputs_label(use_images)

    task = (
        "Now, given the {} between social agent (Jibo) and a participant, "
        "return whether the agent exhibits (A) Social Competence or (B) Social Error."
    ).format(inputs)

    definitions = (
        "We share the definitions here: "
        "(A) Social Competence: Social competence is the ability to successfully conduct social interactions, "
        "which depends on the awareness and identification of social-emotional cues, the ability to process such cues, "
        "and the ability to decide on and express a normative response to these cues. "
        "(B) Social Error: are errors that violate social norms and degrade a user's perception of a robot's socio-affective competence, "
        "such as interrupting a user at an inappropriate time during a conversation. Simply put, socio-affective competence refers to "
        "skillful social and affective behavior that is aligned to the desired and/or normal behaviors expected by a user, thereby increasing trust, reliability, "
        "and overall perceived competence of the agent. Socio-affective error refers to a behavior exhibited by a robot that deviates from the desired or normal "
        "behaviors expected by a user, thereby degrading the overall perceived competence of the agent."
    )

    answer_choices = (
        "\n\nChoose from the following:\n"
        "(A) Social Competence\n"
        "(B) Social Error"
    )

    context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(transcript)

    example_answer = (
        "\n\nThe answer should be in following format:\n"
        "Answer: A"
    )

    return prefix + task + "\n\n" + definitions + answer_choices + context_prompt + example_answer


def build_attribute_prompt(task_type: str, transcript: str, sample: dict, use_images: bool) -> str:
    prefix = build_task_prefix(task_type)
    inputs = build_inputs_label(use_images)
    social_label = infer_social_label(sample)

    task = (
        "Now, given the {} between social agent (Jibo) and a participant, "
        "return which social attribute categories best explain why the interaction is labeled as {}."
    ).format(inputs, social_label)

    definitions = (
        "We share the definitions here: "
        "(A) Emotions: recognizing, interpreting, or responding to the participant's emotional state. "
        "(B) Engagement: recognizing whether the participant is interested, attentive, participating, disengaged, or leaving the interaction. "
        "(C) Conversational Mechanics: managing turn-taking, timing, interruptions, silence, response delay, topic flow, or conversation repair. "
        "(D) Knowledge State: tracking what Jibo or the participant knows, remembers, believes, has already said, or still needs to know. "
        "(E) User Intention: understanding the participant's goal, request, instruction, readiness, refusal, or intended meaning. "
        "(F) Social Context and Relationships: understanding roles, relationship dynamics, rapport, support, trust, or the social purpose of the interaction. "
        "(G) Social Norms and Routines: following expected politeness, apologies, greetings, closings, norms, rituals, or socially appropriate behavior."
    )

    answer_choices = (
        "\n\nChoose from the following:\n"
        "(A) Emotions\n"
        "(B) Engagement\n"
        "(C) Conversational Mechanics\n"
        "(D) Knowledge State\n"
        "(E) User Intention\n"
        "(F) Social Context and Relationships\n"
        "(G) Social Norms and Routines"
    )

    context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(transcript)

    example_answer = (
        "\n\nThe answer should be in following format:\n"
        "Answer: A,C,F"
    )

    return prefix + task + "\n\n" + definitions + answer_choices + context_prompt + example_answer


def build_rationale_error_prompt(task_type: str, transcript: str, choices: List[str], use_images: bool) -> str:
    prefix = build_task_prefix(task_type)
    inputs = build_inputs_label(use_images)
    reasons_block = _build_choice_block(choices, "Choose from the following reasons")

    task = (
        "Now, given the {} between social agent (Jibo) and a participant, "
        "the interaction has already been labeled as Social Error. "
        "Return which reason best explains why the agent's behavior is a Social Error."
    ).format(inputs)

    definitions = (
        "We share the definition here: "
        "Social Error refers to a behavior exhibited by a robot that deviates from the desired or normal behaviors expected by a user, "
        "thereby degrading the overall perceived competence of the agent."
    )

    context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(transcript)

    example_answer = (
        "\n\nThe answer should be in following format:\n"
        "Answer: (1)"
    )

    return (
        prefix
        + task
        + "\n\n"
        + definitions
        + "\n\n"
        + reasons_block
        + context_prompt
        + example_answer
    )


def build_rationale_competence_prompt(task_type: str, transcript: str, choices: List[str], use_images: bool) -> str:
    prefix = build_task_prefix(task_type)
    inputs = build_inputs_label(use_images)
    reasons_block = _build_choice_block(choices, "Choose from the following reasons")

    task = (
        "Now, given the {} between social agent (Jibo) and a participant, "
        "the interaction has already been labeled as Social Competence. "
        "Return which reason best explains why the agent's behavior is Social Competence."
    ).format(inputs)

    definitions = (
        "We share the definition here: "
        "Social Competence is the ability to successfully conduct social interactions, "
        "which depends on the awareness and identification of social-emotional cues, the ability to process such cues, "
        "and the ability to decide on and express a normative response to these cues."
    )

    context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(transcript)

    example_answer = (
        "\n\nThe answer should be in following format:\n"
        "Answer: (1)"
    )

    return (
        prefix
        + task
        + "\n\n"
        + definitions
        + "\n\n"
        + reasons_block
        + context_prompt
        + example_answer
    )


def build_correction_prompt(task_type: str, transcript: str, choices: List[str], use_images: bool) -> str:
    prefix = build_task_prefix(task_type)
    inputs = build_inputs_label(use_images)
    behaviors_block = _build_choice_block(choices, "Choose from the following behaviors")

    task = (
        "Now, given the {} between social agent (Jibo) and a participant, "
        "the interaction has already been labeled as Social Error. "
        "Return which behavior the agent should have done instead."
    ).format(inputs)

    definitions = (
        "We share the definition here: "
        "Social Error refers to a behavior exhibited by a robot that deviates from the desired or normal behaviors expected by a user, "
        "thereby degrading the overall perceived competence of the agent. "
        "Choose the behavior that would best repair or improve the interaction."
    )

    context_prompt = "\n\nAnswer the above from the following Conversation History:\n{}".format(transcript)

    example_answer = (
        "\n\nThe answer should be in following format:\n"
        "Answer: (1)"
    )

    return (
        prefix
        + task
        + "\n\n"
        + definitions
        + "\n\n"
        + behaviors_block
        + context_prompt
        + example_answer
    )