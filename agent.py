# from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI

class Agent:
    def __init__(self, api_key, provider, model="gpt-3.5-turbo", temperature=1):
        breakpoint()
        if provider == "OpenAI":
            self.llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model="gpt-3.5-turbo")
        elif provider == "Llama":
            llama = LlamaAPI(os.environ['LLAMA_API_TOKEN'])
            self.llm = ChatLlamaAPI(client=llama, temperature=temperature)
