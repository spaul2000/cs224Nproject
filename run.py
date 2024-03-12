import argparse
from math_task import MATH
from trivia_task import TRIVIA
import pandas as pd
import os 

os.environ['LLAMA_API_TOKEN'] = 'LL-S38sNFyBFJMraCD4N5llAbj6hCBLutze0DD24KNGCSWkdRTz5izQJIk57tFbRDLd'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAy9PG3kVjWnBtgbDROGtRqYUh1zxm7-RU'
os.environ['OPENAI_API_KEY'] = ''

def run_task(dataset, num_agents):
    if dataset == 'MATH':
        task = MATH(num_agents=num_agents, model_type='Llama', temperature=1)
    
    data = task.get_question_data('data/math_subset_20.json')
    total_record = []

    for i, d in enumerate(data):
        print(i)
        ensamble_answers= task.prompt_agents(d)
        final_answer = task.get_majority_voting_answer(ensamble_answers)
        result_dict = {"ensamble_answers": ensamble_answers, 'final_answer':final_answer}
        one_record = {}
        for k, v in d.items():
            one_record[k] = v
        for k, v in result_dict.items():
            if isinstance(v, list):
                for j, sub_v in enumerate(v):
                    new_k = k + f"_{j}"
                    one_record[new_k] = sub_v
            else:
                one_record[k] = v
        total_record.append(one_record)
        tmp_df = pd.DataFrame(total_record)
        perf = task.evaluation(tmp_df)
        final_answer = result_dict["final_answer"]
        ground_truth = d["ground_truth"]
        print(f"iteration: {i} final_res: {final_answer}, ground_truth: {ground_truth}, perf: {perf}\n")
        print("************************\n")
    df = pd.DataFrame(total_record)
    df.to_csv('test.csv', index=False)
    results = task.evaluation(df)
    print(results)

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