import os
import json
import base64
import requests
import openai
import google.generativeai as genai

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY", ""))


LANGUAGE_ONLY_MODELS = {
    "GPT4o_MINI_Lang",
    "GPT4o_Lang",
    "Llama-3.2-3B",
    "Llama-3.2-3B-Instruct",
}


def is_language_only_model(model_name: str) -> bool:
    return model_name in LANGUAGE_ONLY_MODELS


def build_inputs_label(model_name: str) -> str:
    if is_language_only_model(model_name):
        return "Conversation History"
    return "Images and Conversation History"


def build_base_prompt(task_type: str, model_name: str) -> str:
    inputs = build_inputs_label(model_name)

    wellness_dataset_prompt = (
        "The social robotic agent is designed to be a social positive psychology coach "
        "that delivers interactive positive psychology interventions and provide other "
        "useful skills to build rapport with college students. "
    )
    empathic_dataset_prompt = (
        "The social robotic agent is designed to be a social support companion that "
        "facilitates the exchange of emotionally relevant stories and employs narrative "
        "therapy techniques to enhance feelings of connection and belonging."
    )

    base_prompt = (
        f"You are given the {inputs} between a social robotic agent (Jibo) and a participant. "
        "Answer the following questions about social interactions."
    )

    if "wellness" in task_type:
        return wellness_dataset_prompt + base_prompt
    return empathic_dataset_prompt + base_prompt


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_llm_messages(prompt: str, transcript: str):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
    ]


def call_openai_llm(model_name: str, prompt: str, transcript: str) -> str:
    client = openai.OpenAI()
    messages = build_llm_messages(prompt, transcript)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def call_gemini_llm(model_name: str, prompt: str, transcript: str) -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt, transcript])
    return getattr(response, "text", str(response))


def call_llm(model_name: str, prompt: str, transcript: str) -> str:
    lower_name = model_name.lower()

    if "gpt" in lower_name:
        return call_openai_llm(model_name, prompt, transcript)
    if "gemini" in lower_name:
        return call_gemini_llm(model_name, prompt, transcript)

    raise ValueError(f"Unsupported LLM model: {model_name}")


def save_results(results, output_path: str):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)