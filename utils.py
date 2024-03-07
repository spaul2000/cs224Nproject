def is_final_answer_correct(df_row):
    if df_row["ground_truth"] == df_row["final_answer"]:
        return True
    return False