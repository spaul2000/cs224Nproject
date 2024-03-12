MATH_TASK_SYSTEM_PROMPT = "Imagine you are an expert skilled in solving mathematical problems and are confident in your answer and often persuades other agents to believe in you. Please keep this in mind."

TRIVIA_TASK_SYSTEM_PROMPT = "Imagine you are an expert skilled in reading comprehension and are confident in your answer and often persuades other agents to believe in you. Please keep this in mind."

prompts = {
    
    "math":{
        "question": "Here is a math problem written in LaTeX:{}\nPlease carefully consider it and explain your reasoning. Put your answer in the form \\boxed{{answer}}, at the end of your response.",
    },
    "triviaQA": { 
        "question": "Using the evidence provided below: \n{} \nPlease answer the following question as accurately as possible. \
            Please only include the answer itself in your response, nothing else. Do not answer in a complete sentence. This is very important! {}"
    }
    
}