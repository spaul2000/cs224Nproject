# from openai import OpenAI
from langchain.chat_models import ChatOpenAI

class Agent:
    def __init__(self, api_key, provider, model="gpt-3.5-turbo", temperature=1):
        if provider == "OpenAI":
            self.llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model="gpt-3.5-turbo")