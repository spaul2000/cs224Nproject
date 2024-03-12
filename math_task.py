from ensemble import AgentEnsemble
from math_equivalance import is_equiv
import utils

import json
from prompts import prompts, MATH_TASK_SYSTEM_PROMPT
import re
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage

class MATH():
    def __init__(self, ensemble_dict, temperature=1):
        self.ensemble = AgentEnsemble(ensemble_dict, temperature)
    
    def get_question_data(self, dataset_path):
        qa_list = []
        data_set = json.load(open(dataset_path))
        for level in data_set.keys():
            for category in data_set[level].keys():
                for problem in data_set[level][category]:
                    
                    solution = problem["solution"]
                    solution = self.math_ans_parser(solution)
                    p = problem['problem']
                  
                    
                    question_prompt = prompts["math"]["question"].format(problem["problem"])
                    question_data = {
                        "level": level,
                        "category": category,
                        "human_prompt": question_prompt,
                        "full_solution": problem["solution"],
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
    
    def get_majority_voting_answer(self, agent_answers):
        count = len(agent_answers)
        sameAsCount = [0 for i in range(count)]
        for i in range(count):
            j = i + 1
            while j < count:
                if is_equiv(agent_answers[i], agent_answers[j]):
                    sameAsCount[i] += 1
                    sameAsCount[j] += 1
                j += 1
        largestCount = 0
        for i in range(count):
            if sameAsCount[i] > sameAsCount[largestCount]:
                largestCount = i
        return agent_answers[largestCount]

    
    def prompt_agents(self, question):
       
        system_prompt = SystemMessage(content=MATH_TASK_SYSTEM_PROMPT)
        human_prompt = HumanMessage(content=question['human_prompt'])

        messages = ([system_prompt, human_prompt])

        answers = []
        # breakpoint()
        for agent in self.ensemble.agents:
            if agent.provider == 'google':
                messages = [human_prompt]
            try:
                answer = agent.llm(messages).content
                answer = self.math_ans_parser(answer)
            except:
                answer = -1
            answers.append(answer)
        
        
        return answers

    def evaluation(self, df):
        return df.apply(utils.is_final_answer_correct, axis=1).mean()