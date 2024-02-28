from ensamble import AgentEnsamble
import json
from prompts import prompts
import re

class MATH():
    def __init__(self, num_agents, model_type, api_key, temperature=1):
        self.ensamble = AgentEnsamble(num_agents, model_type, api_key, temperature)
    
    def get_question_data(self, dataset_path):
        qa_list = []
        data_set = json.load(open(dataset_path))
        for level in data_set.keys():
            for category in data_set[level].keys():
                for problem in data_set[level][category]:
                    
                    solution = problem["solution"]
                    solution = self.math_ans_parser(solution)
                    p = problem['problem']
                    print(f"problem: {p}, solution: {solution}\n")
                    
                    question_prompt = prompts["math"]["question"].format(problem["problem"])
                    question_data = {
                        "level": level,
                        "category": category,
                        "state": question_prompt,
                        "ground_truth": solution,
                    }
                    qa_list.append(question_data)
        return qa_list
    
    def math_ans_parser(self, answer_text):
    # Find all occurrences of the \boxed{} pattern and extract the content, accounting for nested braces
   
    # # Return the content of the last occurrence, or None if there were no matches
    # if matches:
    #     return matches[-1], True
    # return None, False

        match = re.search(r'\\boxed{(.*)}', answer_text)
        if match:
            return match.group(1)
        else:
            return None