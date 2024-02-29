import argparse
from math_task import MATH


def run_task(dataset, num_agents, api_key='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'):
    if dataset == 'MATH':
        task = MATH(num_agents=1, model_type='OpenAI', api_key=api_key, temperature=1)
    

    data = task.get_question_data('data/math_subset_20.json')
    ensamble_answers, gt_answers = task.prompt_agents(data[:1])
    majority_vote = [task.get_majority_voting_answer(answers) for answers in ensamble_answers]
    print(majority_vote, gt_answers)


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