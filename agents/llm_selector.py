import os
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI



def get_best_llm(task):
    if task == "claim_extraction":
        return ChatMistralAI(model="mistral-tiny", api_key=os.environ["MISTRALAI_API_KEY"])
    elif task == "fact_verification":
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    elif task == "scoring":
        return ChatMistralAI(model="mistral-tiny", api_key=os.environ["MISTRALAI_API_KEY"])
    elif task == "aggregation":
        return ChatOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", model="openrouter/auto")
    else:
        return ChatOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", model="openrouter/auto")
