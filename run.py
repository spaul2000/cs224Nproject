import argparse
from math_task import MATH
from trivia_task import TRIVIA
from legal_task import LEGAL
import pandas as pd
import os 

from metrics import calculate_metrics

os.environ['LLAMA_API_TOKEN'] = 'LL-S38sNFyBFJMraCD4N5llAbj6hCBLutze0DD24KNGCSWkdRTz5izQJIk57tFbRDLd'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAy9PG3kVjWnBtgbDROGtRqYUh1zxm7-RU'
os.environ['OPENAI_API_KEY'] = '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'
os.environ['ANTHROPIC_API_KEY'] = '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'

ENSEMBLE = {
    'OpenAI': 0,
    'Llama': 0,
    'google': 0,
    'anthropic': 1
} #options: OpenAI, Llama, google

def run_task(dataset, ensemble_dict=ENSEMBLE):
    if dataset == 'MATH' or 'LEGAL':
        if dataset == 'MATH':
            task = MATH(ensemble_dict=ensemble_dict, temperature=1)
            data = task.get_question_data('data/math_subset_20.json')
        elif dataset == 'LEGAL':
            task = LEGAL(ensemble_dict=ensemble_dict, temperature=1)
            data = task.get_question_data('abercrombie')

    
        
        total_record = []

        for i, d in enumerate(data):
            print(i)
            ensemble_answers= task.prompt_agents(d)
            final_answer = task.get_majority_voting_answer(ensemble_answers)
            result_dict = {"ensemble_answers": ensemble_answers, 'final_answer':final_answer}
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

        calculate_metrics(df, len(task.ensemble.agents))

        df.to_csv('test.csv', index=False)
        results = task.evaluation(df)
        print(results)
    if dataset == 'TRIVIA':
        task = TRIVIA(ensemble_dict=ensemble_dict, temperature=1)
        data = task.get_question_data('data/triviaQA/qa/web-train.json')
        questions = []
        final_preds = []
        i = 0
        for q in data[:100]:
            try:
                ensemble_answers = task.prompt_agents(q)
                final_pred = task.get_majority_voting_answer(ensemble_answers)
                final_preds.append(final_pred)
                questions.append(q)
                i += 1
                print("question ", i)
            except:
                pass
        task.evaluate(questions, final_preds)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--dataset", type=str)

    # Parse the command line arguments
    args = parser.parse_args()


    # Process the dataset with the specified number of agents
    run_task(args.dataset)