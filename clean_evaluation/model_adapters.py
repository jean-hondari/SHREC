import os
import traceback
from functools import lru_cache
from typing import List, Optional, Tuple

import openai
import google.generativeai as genai

openai.api_key = os.environ.get("OPENAI_API_KEY", "")
genai.configure(api_key=os.environ.get("GOOGLE_GENAI_API_KEY", ""))


CLOSED_SOURCE_LLM = {"gpt-4o-mini"}
CLOSED_SOURCE_VLM = {"gpt-4o-mini-vision"}
OPEN_SOURCE_LLM = {"llama"}
OPEN_SOURCE_VLM = {"internvl"}


def normalize_model_family(model_name: str) -> str:
    lower = model_name.lower()
    if lower in CLOSED_SOURCE_LLM:
        return "closed_source_llm"
    if lower in CLOSED_SOURCE_VLM:
        return "closed_source_vlm"
    if lower in OPEN_SOURCE_LLM:
        return "open_source_llm"
    if lower in OPEN_SOURCE_VLM:
        return "open_source_vlm"
    raise ValueError(f"Unsupported model: {model_name}")


def call_openai_llm(prompt: str, transcript: str, temperature: float, seed: Optional[int]) -> str:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=temperature,
        seed=seed,
    )
    return response.choices[0].message.content


def call_openai_vlm(prompt: str, transcript: str, image_paths: List[str], temperature: float, seed: Optional[int]) -> str:
    import base64

    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    content = [{"type": "text", "text": f"{prompt}\n\nConversation History:\n{transcript}"}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
            }
        )

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        seed=seed,
    )
    return response.choices[0].message.content


@lru_cache(maxsize=1)
def get_internvl_model():
    """
    Cache the InternVL model so it is loaded once per process.
    This is important for resumable long-running evaluation.
    """
    from vlmeval.config import supported_VLM

    # This matches the existing repo usage in main_vlm_exp.py
    return supported_VLM["InternVL2-8B"]()


def call_internvl_with_vlmeval(prompt: str, transcript: str, image_paths: List[str]) -> str:
    if not image_paths:
        raise ValueError("InternVL requires at least one image path.")

    model = get_internvl_model()
    query = f"{prompt}\n\nConversation History:\n{transcript}"

    # This is the expected call style used by many vlmeval wrappers.
    return model.generate(image_paths, query)


def call_llama(prompt: str, transcript: str) -> str:
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        device_map="auto",
    )
    merged_prompt = f"{prompt}\n\nConversation History:\n{transcript}"
    result = pipe(
        merged_prompt,
        max_new_tokens=512,
        do_sample=False,
    )
    return result[0]["generated_text"]


def run_model(
    model_name: str,
    prompt: str,
    transcript: str,
    image_paths: Optional[List[str]] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[str, Optional[dict]]:
    family = normalize_model_family(model_name)
    image_paths = image_paths or []

    try:
        if family == "closed_source_llm":
            response = call_openai_llm(prompt, transcript, temperature, seed)
        elif family == "closed_source_vlm":
            response = call_openai_vlm(prompt, transcript, image_paths, temperature, seed)
        elif family == "open_source_vlm":
            response = call_internvl_with_vlmeval(prompt, transcript, image_paths)
        elif family == "open_source_llm":
            response = call_llama(prompt, transcript)
        else:
            raise ValueError(f"Unsupported family: {family}")

        return response, None

    except Exception as e:
        msg = str(e).lower()
        error_payload = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "model_family": family,
        }

        if "out of memory" in msg or "cuda" in msg:
            error_payload["category"] = "cuda_oom_or_cuda_error"
        elif "rate limit" in msg:
            error_payload["category"] = "rate_limit"
        elif "api" in msg or "openai" in msg or "google" in msg:
            error_payload["category"] = "api_error"
        elif "vlmeval" in msg or "internvl" in msg:
            error_payload["category"] = "vlmeval_or_internvl_error"
        else:
            error_payload["category"] = "other"

        return "", error_payload