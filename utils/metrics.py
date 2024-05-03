def calculate_metrics(df, num_agents):
    # num_agents = len(self.ensemble.agents)
    ensemble_columns = [col for col in df.columns if col.startswith("ensemble_answer")]
    ground_truth = df["ground_truth"]

    metrics = {}

    # Percent of time all agents are correct
    all_correct = df[ensemble_columns].apply(lambda x: (x == df.loc[x.name, 'ground_truth']).all(), axis=1)

    # Percent of time all agents are incorrect
    all_incorrect = df[ensemble_columns].apply(lambda x: (x != df.loc[x.name, 'ground_truth']).all(), axis=1)
    
    # Percent of time at least one agent is correct
    at_least_one_correct = df[ensemble_columns].apply(lambda x: (x == df.loc[x.name, 'ground_truth']).any(), axis=1)

    # Percent of time at least one agent is incorrect
    at_least_one_incorrect = df[ensemble_columns].apply(lambda x: (x != df.loc[x.name, 'ground_truth']).any(), axis=1)

    # Percent of time at least one agent is correct and final answer is incorrect
    one_correct_final_incorrect = at_least_one_correct & (df['final_answer'] != ground_truth)

    # Percent of time all agents agree
    all_agree = df[ensemble_columns].apply(lambda x: x.nunique() == 1, axis=1)

    # Percent of time all agents disagree
    all_disagree = df[ensemble_columns].apply(lambda x: x.nunique == num_agents, axis=1)

    print("% of time all agent answers are correct for a given question: ", all_correct.mean())
    print("% of time all agent answers are incorrect for a given question: ", all_incorrect.mean())
    print("% of time at least one agent is correct for a given question: ", at_least_one_correct.mean())
    print("% of time at least agent is incorrect for a given question: ", at_least_one_incorrect.mean())
    print("% of time at least one agent is correct but the final answer is incorrect for a given question: ", one_correct_final_incorrect.mean())
    print("% of time all of the agents agree: ", all_agree.mean())
    print("% of time all agents disagree: ", all_disagree.mean())