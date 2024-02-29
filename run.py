import argparse
from math_task import MATH
from trivia_task import TRIVIA
import pandas as pd

def run_task(dataset, num_agents, api_key='xxx'):
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

    if dataset == 'TRIVIA':
        task = TRIVIA(num_agents=1, model_type='OpenAI', api_key=api_key, temperature=1)
        data = task.get_question_data('data/triviaQA/qa/web-train.json')
        questions = []
        final_preds = []
        
        i = 0
        for q in data:
            if i == 100:
                break
            try:
                answers = task.prompt_agents(q)
                final_pred = task.get_majority_voting_answer(answers)
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
    parser.add_argument("--num_agents", type=int)
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Process the dataset with the specified number of agents
    run_task(args.dataset, args.num_agents)