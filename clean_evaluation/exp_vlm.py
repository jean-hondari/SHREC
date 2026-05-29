import os
import json
import base64
import openai
import google.generativeai as genai

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY", ""))


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_openai_vision_messages(prompt: str, transcript: str, image_paths):
    content = [{"type": "text", "text": f"{prompt}\n\nConversation History:\n{transcript}"}]

    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    return [{"role": "user", "content": content}]


def call_openai_vlm(model_name: str, prompt: str, transcript: str, image_paths) -> str:
    client = openai.OpenAI()
    messages = build_openai_vision_messages(prompt, transcript, image_paths)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def call_gemini_vlm(model_name: str, prompt: str, transcript: str, image_paths) -> str:
    model = genai.GenerativeModel(model_name)

    content = [prompt, f"Conversation History:\n{transcript}"]
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        content.append(
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            }
        )

    response = model.generate_content(content)
    return getattr(response, "text", str(response))


def call_local_vlm(model_name: str, prompt: str, transcript: str, image_paths) -> str:
    from vlmeval.config import supported_VLM

    if model_name == "paligemma":
        model = supported_VLM["paligemma-3b-mix-448"]()
    elif model_name == "llava_next_llama3":
        model = supported_VLM["llava_next_llama3"]()
    else:
        raise ValueError(f"Unsupported local VLM model: {model_name}")

    query = f"{prompt}\n\nConversation History:\n{transcript}"
    response = model.generate(image_paths, query)
    return response


def call_vlm(model_name: str, prompt: str, transcript: str, image_paths) -> str:
    lower_name = model_name.lower()

    if model_name in {"paligemma", "llava_next_llama3"}:
        return call_local_vlm(model_name, prompt, transcript, image_paths)
    if "gpt" in lower_name:
        return call_openai_vlm(model_name, prompt, transcript, image_paths)
    if "gemini" in lower_name:
        return call_gemini_vlm(model_name, prompt, transcript, image_paths)

    raise ValueError(f"Unsupported VLM model: {model_name}")


def save_results(results, output_path: str):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)