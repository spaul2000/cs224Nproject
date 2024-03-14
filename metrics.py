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

    print("ALL CORRECT")
    print(all_correct.mean())
    print("ALL INCORRECT")
    print(all_incorrect.mean())
    print("AT LEAST ONE CORRECT")
    print(at_least_one_correct.mean())
    print("AT LEAST ONE INCORRECT")
    print(at_least_one_incorrect.mean())
    print("ONE CORRECT FINAL INCORRECT")
    print(one_correct_final_incorrect.mean())
    print("ALL AGREE")
    print(all_agree.mean())
    print("ALL DISAGREE")
    print(all_disagree.mean())