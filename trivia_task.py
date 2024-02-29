from ensamble import AgentEnsamble
import json
from prompts import prompts
import re

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
            answer = data['Answer']['Value']
            normalized_answer =  data['Answer']['NormalizedValue']
            aliases = data['Answer']['Aliases']
            normalized_aliases = data['Answer']['NormalizedAliases']
            question_data = {
                "question": question,
                "evidence": evidence,
                "answer": answer,
                "normalized_answer": normalized_answer,
                "aliases": aliases,
                "normalized_aliases": normalized_aliases
            }
            qa_list.append(question_data)
        return qa_list

