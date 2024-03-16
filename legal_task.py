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
import json


from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class LEGAL():
    def __init__(self, ensemble_dict, temperature=1, task_name="abercrombie"):
        self.task_name = task_name
        self.ensemble = AgentEnsemble(ensemble_dict, temperature)
        self.labels = []
        self.times = []

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
    
    
    # def parse_answer(self, answer_list):
    #     """Parses a model output and extracts the last relevant answer label.

    #     Args:
    #         answer: A list containing model outputs.

    #     Returns:
    #         A final answer label.
    #     """

    #     labels = self.labels
    #     parsed_answers = []

    #     for answer_str in answer_list:
    #         for label in labels:  # Iterate through labels in reverse order
    #             if label in answer_str.lower():  
    #                 # breakpoint()
    #                 parsed_answers.append(label)
    #                 break  # Move on to the next answer_str 
    #         else:
    #             parsed_answers.append("None")
        
    #     return parsed_answers
    
    def parse_answer(self, answer):

        labels = self.labels
        for string in reversed(answer.lower().split()):
            if string in labels:
                return string
        
        return "None"

    def get_majority_voting_answer(self, agent_answers):
        count = len(agent_answers)
        # breakpoint()
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
        # breakpoint()
        return agent_answers[maxIndex]

    def prompt_agents(self, question):
        start_time = time.time()
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

       
        # print("PRE-PARSED ANSWERS")
        print(answers)

        answers = self.parse_answer(answers)
        
        print("ANSWERS")
        print(answers)
        finish_time = time.time()
        total_time = finish_time - start_time
        print(f"Time taken to get answers: {total_time} seconds")
        self.times.append(total_time)
        return answers


    def prompt_agents_threadpool(self, question):
        start_time = time.time()
        def invoke_agent(agent, prompt):
            return agent.llm.invoke(prompt).content

        system_prompt = SystemMessage(content=LEGAL_TASK_SYSTEM_PROMPT)
        human_prompt = HumanMessage(content=question['human_prompt'])
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format()
        
        answers = []
        # Set the number of threads to the number of agents, or another suitable number
        # with ThreadPoolExecutor(max_workers=len(self.ensemble.agents)) as executor:
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit all API calls to the executor
            future_to_agent = {executor.submit(invoke_agent, agent, prompt): agent for agent in self.ensemble.agents}
            
            # Retrieve the results as they are completed
            for future in as_completed(future_to_agent):
                answers.append(future.result())
                print("Answer Received")

        answers = self.parse_answer(answers)
        finish_time = time.time()
        total_time = finish_time - start_time
        print(f"Time taken to get answers: {total_time} seconds")
        self.times.append(total_time)
        return answers


    def prompt_agents_advanced_threadpool(self, questions):
        questions = questions  # Limit to first 10 questions for testing

        # Define the column names
        num_agents = len(self.ensemble.agents)
        columns = ['human_prompt', 'ground_truth'] + [f'ensemble_answers_{i}' for i in range(num_agents)] + ['final_answer']
        num_rows = len(questions)  # Number of questions
        df = pd.DataFrame(columns=columns, index=range(num_rows))  # Create an empty DataFrame

        start_time = time.time()

        # def invoke_agent(agent, prompt, question_index):
        #     # Include question_index to track which question this answer belongs to
        #     answer = agent.llm.invoke(prompt).content
        #     return agent, self.parse_answer(answer), question_index

        def invoke_agent(agent, prompt, question_index, max_retries=3):
            try:
                # Make the API call and get AIMessage object
                ai_message = agent.llm.invoke(prompt)
                
                # Assuming the AIMessage class has 'content' attribute to hold the response
                answer = ai_message.content
                parsed_answer = self.parse_answer(answer)
                
                return agent, parsed_answer, question_index

            except Exception as e:
                # Handle exceptions and errors
                print(f"Error in invoke_agent: {str(e)}")
                print(f"Retrying... {max_retries} attempts left")
                if max_retries > 0:
                    return invoke_agent(agent, prompt, question_index, max_retries - 1)
                else:
                    return agent, "Error processing response", question_index


        unlimited_agents = [agent for agent in self.ensemble.agents if agent.provider != "google"]
        # unlimited_agents = [agent for agent in self.ensemble.agents]
        limited_agents = [agent for agent in self.ensemble.agents if agent.provider == "google"]
        # limited_agents = []

        # Prepare the prompts for all questions in advance
        prompts = []
        for question in questions:
            system_prompt = SystemMessage(content=LEGAL_TASK_SYSTEM_PROMPT)
            human_prompt = HumanMessage(content=question['human_prompt'])
            prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt]).format()
            prompts.append(prompt)

        # Submit all jobs for all questions to the threadpool
        # with ThreadPoolExecutor(max_workers=len(unlimited_agents) * 3) as executor:  # Control the number of workers here
        with ThreadPoolExecutor(max_workers=10) as executor:  # Control the number of workers here
            futures = []
            for i, prompt in enumerate(prompts):
                for agent in unlimited_agents:
                    futures.append(executor.submit(invoke_agent, agent, prompt, i))

            # Process the results as they complete
            for future in as_completed(futures):
                print("Answer Received Threadpool")
                agent, answer, question_index = future.result()
                agent_index = self.ensemble.agents.index(agent)
                df.loc[question_index, f'ensemble_answers_{agent_index}'] = answer

        # Process limited agents sequentially for each question
        for i, prompt in enumerate(prompts):
            for agent in limited_agents:
                print("Answer Received Sequential")
                agent_index = self.ensemble.agents.index(agent)
                answer = invoke_agent(agent, prompt, i)[1]  # We already know the question index here
                df.loc[i, f'ensemble_answers_{agent_index}'] = answer

        # Fill in human_prompt and ground_truth for each question
        for i, question in enumerate(questions):
            df.loc[i, 'human_prompt'] = question['human_prompt']
            df.loc[i, 'ground_truth'] = question['ground_truth']

        # Get majority vote for each question
        df['final_answer'] = df.apply(lambda x: self.get_majority_voting_answer(x[2:num_agents+2]), axis=1)

        # Log time
        finish_time = time.time()
        total_time = finish_time - start_time
        print(f"Time taken to get answers: {total_time} seconds")
        
        return df
 
    def evaluation(self, df):
        # return legalbench_evaluation.evaluation(df)
        # return evaluation.evaluation(df)
        generations = df["final_answer"].to_list()
        ground_truth = df["ground_truth"].to_list()

        # print("GENERATIONS")
        # print(generations)
        # print("GROUND TRUTH")
        # print(ground_truth)

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
