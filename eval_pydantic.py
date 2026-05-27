from pydantic import BaseModel
import openai
from openai import OpenAI
import os 
import json
import pdb 
import time
from tqdm import tqdm
import pdb
import argparse
openai.api_key=os.environ["OPENAI_API_KEY"]
client = OpenAI()

class Event(BaseModel):
    answer: list[str]
    explanation: list[str]


def parse_answer(sample, task):


    if 'detection' in task:
        system_prompt =  "Given the [ANSWER CHOICES], extract the answer from the [RESPONSE] among the following options, if the answer does not exist, output 'Does not exist' : (A) Social Competence, (B) Social Error, (C) None"

    if 'attribute' in task:
        system_prompt = "Given the [ANSWER CHOICES], extract the answer from the [RESPONSE] among the following options, there can be multiple. If the answer does not exist, output 'Does not exist' : (A) Recognizing Emotions (B) Recognizing Engagement (C) Recognition of Conversational Mechanics (D) Understanding Knowledge State of Others and Self (E) Understanding Intention of Others (F) Recognizing Social Relationships (G) Recognizing Social Norms including toxicity"
    
    if 'rationale' in task or 'correction'in task or 'pre' in task or 'post' in task :
        system_prompt = "Given the [ANSWER CHOICES], extract the answer from the [RESPONSE] among the following options, if the answer does not exist, output 'Does not exist' : (1), (2), (3), (4), (5)"

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample},
        ],
        response_format=Event,
    )
    return completion

def process_file(input_file, output_directory):
    filename = os.path.basename(input_file)
    output_result_path = os.path.join(output_directory, filename)

    if os.path.exists(output_result_path):
        print(f"Skipping already processed file: {filename}")
        return

    print(f"Processing: {filename}")

    with open(input_file, "r") as file:
        data = json.load(file)

    curr_exp_result = []

    for sample in tqdm(data, desc=f"Parsing {filename}"):
        try:
            question = "[ANSWER CHOICES]: {} \n ".format(sample["question"].split("\n\n")[-1])
            response = "[RESPONSE]: {}".format(sample["response"])

            completion = parse_answer(question + response, filename)
            event = completion.choices[0].message.parsed

            updated_sample = dict(sample)
            updated_sample["pre-response"] = sample["response"]
            updated_sample["response"] = event.answer
            updated_sample["val_explanation"] = event.explanation
            curr_exp_result.append(updated_sample)

        except Exception as e:
            failed_sample = dict(sample)
            failed_sample["pre-response"] = sample.get("response", None)
            failed_sample["response"] = ["PARSE_ERROR"]
            failed_sample["val_explanation"] = [str(e)]
            curr_exp_result.append(failed_sample)

    with open(output_result_path, "w") as f:
        json.dump(curr_exp_result, f)

    print(f"Saved: {output_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./output",
        help="Path to a single JSON file or a directory of JSON files."
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./output_pydantic",
        help="Directory where parsed outputs will be saved."
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_directory = args.output_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    start = time.time()

    if os.path.isfile(input_path):
        process_file(input_path, output_directory)

    elif os.path.isdir(input_path):
        for filename in sorted(os.listdir(input_path)):
            if not filename.endswith(".json"):
                continue

            input_file = os.path.join(input_path, filename)
            process_file(input_file, output_directory)
    else:
        raise ValueError(f"Input path does not exist or is invalid: {input_path}")

    end = time.time()
    print(end - start)