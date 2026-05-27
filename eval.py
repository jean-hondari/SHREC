import os 
import json
import pdb
from natsort import natsorted



dataset = 'empathic'
dir_path = "./output_pydantic"
key_dict = {"A": "Emotions", "B": "Engagement", "C": "Conversational Mechanics", "D": "Knowledge State", "E":"Intention", "F":"Social Context and Relationships", "G":"Social Norms and Routines"}



exps = os.listdir(dir_path)
string_list = []
for exp in exps: 
        
    exp_list = exp.split("_")
    

    if dataset not in exp:
        continue

    model = exp.split(dataset)[0]
    task = exp.split(dataset)[1]
    
    result_path = os.path.join(dir_path, exp)
    with open(result_path, 'r') as file:
        data = json.load(file)




    if len(data) == 0:
        continue

    correct_count = 0
    for sample in data: 

        if len(sample['response']) == 0:
            continue

        
        if 'pydantic' in dir_path:
            answer= sample['response'][0]
        else:
            answer = sample['response']
  
        answer_list = sample['response']
        if 'detection' in task:
            if 'social competence' in answer.lower() or 'A' in answer:
                if sample['label'] == False:
                    correct_count += 1 
            elif 'social error' in answer.lower() or 'B' in answer:
                if sample['label'] == True:
                    correct_count += 1
            elif 'C' in answer or 'none' in answer.lower(): 
                if isinstance(sample['label'], dict) and sample['label'].get('isCompetence') is None:
                    correct_count += 1 
        
        if 'attribute' in task:
            
            
            answer_k = None
            answer_v = None
            for k,v in key_dict.items():
                if k in answer:
                    answer_k = k
                    answer_v = v

            if answer_v == None:
                continue 
                
            
            if sample['label'][answer_v]: 

                correct_count += 1 


        if 'attribute_agreed_multiple'in exp or 'attribute_disagree' in exp:
            inv_key_dict = {v: k for k, v in key_dict.items()}
        
            answer_k = None
            answer_v = None


            correct = True 
            for k,v in sample['label'].items():
                if v == True:
                    a_b_c = inv_key_dict[k]
                    if a_b_c in answer_list: 
                        pass 
                    else:
                        correct = False 
            
                if v == False:
                    a_b_c = inv_key_dict[k]
                    if a_b_c in answer_list: 
                        correct = False 
                    else:
                        pass

            correct_count += int(correct)
                    
        if 'post' in task or 'pre' in task or 'rationale' in task or'correction' in task:
            final_find_parenth = None
            for answer_v in ["1", "2", "3", "4", "5"]:
                find_parenth = answer.find(answer_v)
                if not find_parenth == -1:
                    final_find_parenth = answer_v
                    
            if final_find_parenth == None:
                continue

            if final_find_parenth in sample['label']:
                correct_count += 1 

    accuracy = round(correct_count/len(data),3)
    res_string = "Accuracy for {}: {}".format(exp, accuracy)
    
    print(res_string)