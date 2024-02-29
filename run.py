import argparse
from math_task import MATH
import pandas as pd

def run_task(dataset, num_agents, api_key='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'):
    if dataset == 'MATH':
        task = MATH(num_agents=num_agents, model_type='OpenAI', api_key=api_key, temperature=1)
    

    data = task.get_question_data('data/math_subset_20.json')
    total_record = []

    for i, d in enumerate(data[:2]):

        ensamble_answers= task.prompt_agents(d)
        final_answer = task.get_majority_voting_answer(ensamble_answers)
        result_dict = {"ensamble_answers": ensamble_answers, 'final_answer':final_answer}
        one_record = {}
        for k, v in d.items():
            one_record[k] = v
        for k, v in result_dict.items():
            if isinstance(v, list):
                for i, sub_v in enumerate(v):
                    new_k = k + f"_{i}"
                    one_record[new_k] = sub_v
            else:
                one_record[k] = v
        total_record.append(one_record)
        df = pd.DataFrame(total_record)
        df.to_csv('test.csv', index=False)
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_agents", type=int)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Process the dataset with the specified number of agents
    run_task(args.dataset, args.num_agents)