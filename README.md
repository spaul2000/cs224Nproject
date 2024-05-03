# Multi-Agent Frameworks in Domain-Specific Question Answering Tasks

This repository contains the code used to generate experiments for our paper "Multi-Agent Frameworks in Domain-Specific Question Answering Tasks." For detailed insights and results, refer to the `CS224N_Project_Final_Report.pdf`.



### Prerequisites
Ensure you have set the environment variables for the OpenAI, Llama, Google, and Anthropic API keys to run the code.

### Configuration
Modify the `ENSEMBLE` configuration in the `run.py` file to specify the quantity and type of language models (LLMs) for the ensemble. Available options include OpenAI, Llama, Google, and Anthropic:

```python
ENSEMBLE = {
    'OpenAI': 4,
    'Llama': 4,
    'Google': 4,
    'Anthropic': 0
}
```

### Usage 

To run experiments, use the following command with the dataset options MATH, LEGAL, or TRIVIA to specify which type of task you would like to evaluate the LLM ensemble on:

`python run.py --dataset MATH LEGAL TRIVIA`


### Repository Structure

- **/tasks**: 
  - Contains code to execute language model (LLM) ensembles on each of the specified task categories.

- **/llm**: 
  - Includes code for generating and managing LLM agent ensembles.

- **/utils**: 
  - Houses utilities for prompting, evaluation, and answer formatting. This includes specialized code for handling mathematical instances.
