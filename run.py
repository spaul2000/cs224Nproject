import argparse
from math_task import MATH
def run_task(dataset, num_agents, api_key='xxx'):
    if dataset == 'MATH':
        task = MATH(num_agents=1, model_type='OpenAI', api_key=api_key, temperature=1)
    

    data = task.get_question_data('data/math_subset_20.json')

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