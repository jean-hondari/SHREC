from pydantic import BaseModel
import openai
from openai import OpenAI
import os 
import json
import pdb 
import time
from tqdm import tqdm
import pdb

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



if __name__ == "__main__": 
    directory = "./output"
    output_directory = "./output_pydantic"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    start = time.time()

    for task in ['pre', 'post', 'rationale', 'correct', 'detection', 'attribute']:


        for filename in os.listdir(directory):
            if task in filename:  
                curr_exp_result = [] 

                result_path = os.path.join(directory, filename)

                output_result_path = os.path.join(output_directory, filename)

                if os.path.exists(output_result_path):
                    pass
                    # print('Exists so skipping')

                if not os.path.exists(output_result_path):
                    print(filename)
                    task = filename
                    with open(result_path, 'r') as file:
                        data = json.load(file)

                    sample = data[0]
                    question = "[ANSWER CHOICES]: {} \n ".format(sample['question'].split('\n\n')[-1])

                    response = "[RESPONSE]: {}".format(sample['response']) 
                    completion = parse_answer(question + response, task)
                    event = completion.choices[0].message.parsed
                    sample['pre-response'] = sample['response']
                    sample['response'] = event.answer
                    sample['val_explanation'] = event.explanation
                    curr_exp_result.append(sample)
                    
                    with open(output_result_path, 'w') as f:
                        json.dump(curr_exp_result, f)
            
            
    end = time.time()
    print(end - start)


        