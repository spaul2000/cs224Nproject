# from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_google_genai import ChatGoogleGenerativeAI


class Agent:
    def __init__(self, provider, temperature=1):
        if provider == "OpenAI":
            self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        elif provider == "Llama":
            print('llama')
            llama = LlamaAPI(os.environ['LLAMA_API_TOKEN'])
            self.llm = ChatLlamaAPI(client=llama, temperature=temperature)
        elif provider == "google":
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
        else:
            print("invalid model input")
