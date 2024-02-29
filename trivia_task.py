from ensemble import AgentEnsemble
import json
import csv
import re
import string
from prompts import prompts, TRIVIA_TASK_SYSTEM_PROMPT
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import tiktoken

class TRIVIA():
    def __init__(self, num_agents, model_type, api_key, temperature=1):
        self.ensemble = AgentEnsemble(num_agents, model_type, api_key, temperature)

    def get_question_data(self, dataset_path):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        evidence_path = "data/triviaQA/evidence/web/"
        if "wikipedia" in dataset_path:
            evidence_path = "data/triviaQA/evidence/wikipedia/"
        qa_list = []
        data_set = json.load(open(dataset_path))["Data"]
        i = 0
        for data in data_set:    
            if (i % 1000 == 0):
                print('data: ', i)
            
            question = data['Question']
            search_results = data['SearchResults'] # list of files
            evidence = []
            for result in search_results:
                filename = result['Filename']
                f = open(evidence_path + filename, 'r')
                content = f.read()
                evidence.append(content)
            question_prompt = prompts["triviaQA"]["question"].format('\n\n'.join(evidence), question)
            tokens = encoding.encode(question_prompt)
            token_count = len(tokens)
            if token_count >= 16000:
                continue
            answer = data['Answer']['Value']
            normalized_answer =  data['Answer']['NormalizedValue']
            aliases = data['Answer']['Aliases']
            normalized_aliases = data['Answer']['NormalizedAliases']
            question_id = data['QuestionId']
            question_data = {
                "question": question,
                "human_prompt": question_prompt,
                "evidence": evidence,
                "answer": answer,
                "normalized_answer": normalized_answer,
                "aliases": aliases,
                "normalized_aliases": normalized_aliases,
                "question_id": question_id
            }
            i += 1
            print(i)
            qa_list.append(question_data)
            if i == 200:
                break
        return qa_list
    

    def prompt_agents(self, question):
        # takes in one question dict, runs all agents in ensemble, returns list of answers
        system_prompt = SystemMessagePromptTemplate.from_template(TRIVIA_TASK_SYSTEM_PROMPT)
        human_prompt = HumanMessagePromptTemplate.from_template(question['human_prompt'])

        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format()
        answers = []
        for agent in self.ensemble.agents:
            answer = agent.llm.invoke(prompt)
            answers.append(answer.content)
        return answers

    # def trivia_ans_parser(self, answers):
    #     # takes in list of answers to one question 
    #     print(parsed_answers)
    #     parsed_answers = []
    #     for answer in answers:
    #         parsed_answers.append(answer.content)

    def get_majority_voting_answer(self, agent_answers):
        # for one prompt
        if len(agent_answers) == 1:
            return agent_answers[0]
        

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace('_', ' ')

        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

    def has_exact_match(self, ground_truths, candidates):
        for ground_truth in ground_truths:
            if ground_truth in candidates:
                return True
        return False


    def evaluate(self, questions, answers):
        num_correct = 0
        df = open("trivia_one_agent.csv", 'w', newline='')
        fieldnames = ["question_ids", "possible_answers", "predicted_answer", "normalized_answer", "correct"]
        writer = csv.DictWriter(df, fieldnames=fieldnames)
        for i in range(len(questions)):
            row = {}
            question = questions[i]
            possible_answers = [question["answer"], question["normalized_answer"]] + question["aliases"] + question["normalized_aliases"]
            row["question_ids"] = question["question_id"]
            row["possible_answers"] = possible_answers
            row["predicted_answer"] = answers[i]
            row["correct"] = "False"
            normalized_ans = self.normalize_answer(answers[i])
            row["normalized_answer"] = normalized_ans
            
            if normalized_ans in possible_answers or self.has_exact_match(possible_answers, normalized_ans):
                num_correct += 1
                row["correct"] = "True"
            writer.writerow(row)
        #df.to_csv("trivia_one_agent.csv")
        print("Accuracy: ", num_correct/len(questions))
        return