from ensamble import AgentEnsamble
import json
from prompts import prompts, TRIVIA_TASK_SYSTEM_PROMPT
import re
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

class TRIVIA():
    def __init__(self, num_agents, model_type, api_key, temperature=1):
        self.ensamble = AgentEnsamble(num_agents, model_type, api_key, temperature)

    def get_question_data(self, dataset_path):
        evidence_path = "data/triviaQA/evidence/web/"
        if "wikipedia" in dataset_path:
            evidence_path = "data/triviaQA/evidence/wikipedia/"
        qa_list = []
        data_set = json.load(open(dataset_path))["Data"]
        for data in data_set:    
            question = data['Question']
            search_results = data['SearchResults'] # list of files
            evidence = []
            for result in search_results:
                filename = search_results['Filename']
                f = open(evidence_path + filename, 'r')
                content = f.read()
                evidence.append(content)
            question_prompt = prompts["triviaQA"]["question"].format('\n\n'.join(evidence), question)
            answer = data['Answer']['Value']
            normalized_answer =  data['Answer']['NormalizedValue']
            aliases = data['Answer']['Aliases']
            normalized_aliases = data['Answer']['NormalizedAliases']
            question_data = {
                "question": question,
                "human_prompt": question_prompt,
                "evidence": evidence,
                "answer": answer,
                "normalized_answer": normalized_answer,
                "aliases": aliases,
                "normalized_aliases": normalized_aliases
            }
            qa_list.append(question_data)
        return qa_list
    

    # def trivia_ans_parser(self, answer_text):
    #     pass
    def prompt_agents(self, questions):
        for question in questions:
            system_prompt = SystemMessagePromptTemplate.from_template(TRIVIA_TASK_SYSTEM_PROMPT)
            human_prompt = HumanMessagePromptTemplate.from_template(question['human_prompt'])

            prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            
            answers = []
            for agent in self.ensamble.agents:
                answer = agent.llm(prompt)
                answers.append(answer)
            print(answers)
