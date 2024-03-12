from ensemble2 import AgentEnsemble
import json
import csv
import re
import string
from sacrebleu import sentence_bleu
from prompts import prompts, TRIVIA_TASK_SYSTEM_PROMPT
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import tiktoken

class TRIVIA():
    def __init__(self, num_agents, model_type, api_key, temperature=1):
        self.ensemble = AgentEnsemble(num_agents, api_key, temperature)

    def get_question_data(self, dataset_path):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        evidence_path = "data/triviaQA/evidence/web/"
        if "wikipedia" in dataset_path:
            evidence_path = "data/triviaQA/evidence/wikipedia/"
        qa_list = []
        data_set = json.load(open(dataset_path))["Data"]
        i = 0
        for data in data_set:    
            # if (i % 1000 == 0):
            #     print('data: ', i)
            
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
            qa_list.append(question_data)
            if i == 500:
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
    

    def get_majority_voting_answer(self, agent_answers):
        if len(agent_answers) == 1:
            # for one agent
            return agent_answers[0]
        # more than one agents
        cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score
        bleu_scores = []
        for idx, agent in enumerate(agent_answers):
            total_score = 0
            for idx_o, otheragent in enumerate(agent_answers):
                if idx == idx_o:
                    continue
                total_score += cmp_res(self.normalize_answer(agent), self.normalize_answer(otheragent))
            bleu_scores.append(total_score)
        max_index = max(enumerate(bleu_scores), key=lambda x: x[1])[0]
        return agent_answers[max_index]

    

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


    def has_exact_match(self, ground_truths, candidate):
        print(ground_truths)
        print(self.normalize_answer(candidate))
        for ground_truth in ground_truths:
            if self.normalize_answer(ground_truth) in self.normalize_answer(candidate):
                return True
        return False

    def bleuscore_one_pred(self, possible_answers, pred):
        cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score
        bleu_scores = []
        for poss_ans in possible_answers:
            score = cmp_res(self.normalize_answer(poss_ans), self.normalize_answer(pred))
            bleu_scores.append(score)
        return bleu_scores



    def evaluate(self, questions, answers):
        num_correct = 0
        df = open("trivia_twenty_llama.csv", 'w', newline='')
        fieldnames = ["question_ids", "possible_answers", "predicted_answer", "normalized_answer", "correct", "bleu_scores"]
        writer = csv.DictWriter(df, fieldnames=fieldnames)
        for i in range(len(questions)):
            
            question = questions[i]
            possible_answers = [question["answer"], question["normalized_answer"]] + question["aliases"] + question["normalized_aliases"]
            normalized_ans = self.normalize_answer(answers[i])
            row = {
                "question_ids": question["question_id"],
                "possible_answers": possible_answers,
                "predicted_answer": answers[i],
                "correct": "False",
                "normalized_answer": normalized_ans,
                "bleu_scores": []
            }
            if normalized_ans in possible_answers or self.has_exact_match(possible_answers, normalized_ans):
                num_correct += 1
                row["correct"] = "True"
            else:
                row["bleu_scores"] = self.bleuscore_one_pred(possible_answers, normalized_ans)

            writer.writerow(row)
        #df.to_csv("trivia_one_agent.csv")
        print("Accuracy: ", num_correct/len(questions))
        return
    

