import datasets
import requests

# from ensemble import AgentEnsemble
from ensemble import AgentEnsemble
from LegalBench import evaluation as legalbench_evaluation

from LegalBench.tasks import TASKS, ISSUE_TASKS
from LegalBench.utils import generate_prompts

import utils

from prompts import prompts, LEGAL_TASK_SYSTEM_PROMPT

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_core.messages import HumanMessage, SystemMessage

import pandas as pd

class LEGAL():
    def __init__(self, ensemble_dict, temperature=1, task_name="abercrombie"):
        self.task_name = task_name
        self.ensemble = AgentEnsemble(ensemble_dict, temperature)
        self.labels = []

    def get_question_data(self, dataset_name):
        dataset = datasets.load_dataset("nguha/legalbench", dataset_name)
        train_dataset = dataset["train"].to_pandas()    
        test_dataset = dataset["test"].to_pandas()

        # print(len(train_dataset))
        # print(len(test_dataset))

        unique_labels = train_dataset["answer"].unique()
        self.labels = unique_labels

        options_string = "Your answer must be one of the following: " + ", ".join(unique_labels) + "."

        github_base_url = f"https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{dataset_name}/"  
        base_prompt_file = "base_prompt.txt"
        base_prompt_url = github_base_url + base_prompt_file

        response = requests.get(base_prompt_url)

        if response.status_code == 200:
            prompt_template = response.text
        else:
            print("Error fetching base prompt. Status code:", response.status_code)

        questions = generate_prompts(prompt_template=prompt_template, data_df=test_dataset)
        for i, prompt in enumerate(questions):
            questions[i] = options_string + prompts["legal"]["question"].format(prompt)

        answers = test_dataset["answer"]
        answers = answers.to_list()

        qa_list = []
        for prompt, answer in zip(questions, answers):
            qa_dict = {
                "human_prompt": prompt,
                "ground_truth": answer
            }
            qa_list.append(qa_dict)

        # breakpoint()

        return qa_list
    
    def parse_answer(self, answer_list):
        """Parses a list of model outputs to extract the last relevant answer label.

        Args:
            answer_list: A list containing model outputs.

        Returns:
            A list of final answer labels.
        """

        labels = self.labels
        parsed_answers = []

        for answer_str in answer_list:
            for label in labels:  # Iterate through labels in reverse order
                if label in answer_str.lower():  
                    parsed_answers.append(label)
                    break  # Move on to the next answer_str 
            else:
                parsed_answers.append(answer_str)

        return parsed_answers

    def get_majority_voting_answer(self, agent_answers):
        count = len(agent_answers)
        sameAsCount = [0 for i in range(count)]
        for i in range(count):
            j = i + 1
            while j < count:
                if agent_answers[i] == agent_answers[j]:
                    sameAsCount[i] += 1
                    sameAsCount[j] += 1
                j += 1
        maxIndex = 0
        # maxCount = 0
        for i in range(count):
            if sameAsCount[i] > sameAsCount[maxIndex]:
                maxIndex = i
        return agent_answers[maxIndex]

    def prompt_agents(self, question):
        system_prompt = SystemMessage(content=LEGAL_TASK_SYSTEM_PROMPT)
        human_prompt = HumanMessage(content=question['human_prompt'])

        # messages = [system_prompt, human_prompt]
        # print(messages)
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format()
        answers = []

        for agent in self.ensemble.agents:
            # answer = agent.llm(messages).content
            answer = agent.llm.invoke(prompt)
            answers.append(answer.content)
            # print("ANSWER")
            # print(answer)

       
        print("PRE-PARSED ANSWERS")
        print(answers)

        answers = self.parse_answer(answers)
        
        print("ANSWERS")
        print(answers)

        return answers

    def evaluation(self, df):
        # return legalbench_evaluation.evaluation(df)
        # return evaluation.evaluation(df)
        generations = df["final_answer"].to_list()
        ground_truth = df["ground_truth"].to_list()
        return legalbench_evaluation.evaluate(self.task_name, generations, ground_truth)
        # return df.apply(utils.is_final_answer_correct, axis=1).mean()
    

def main():
    # Create an instance of the LEGAL class
    legal = LEGAL(num_agents=3, model_type="bert", api_key="your_api_key", temperature=1)

    # Test the get_question_data method
    dataset_path = "abercrombie"
    legal.get_question_data(dataset_path)

    # # Test the prompt_agents method
    # question = "What is the legal age to vote?"
    # responses = legal.prompt_agents(question)
    # print(responses)

    # # Test the evaluation method
    # df = pd.DataFrame(...)  # Create a DataFrame with evaluation data
    # result = legal.evaluation(df)
    # print(result)

if __name__ == "__main__":
    main()
