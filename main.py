import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ChatGroq_API_KEY")

id_model = "llama-3.3-70b-versatile"
temperature = 0.7

def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    api_key=api_key,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )

llm = load_llm(id_model, temperature)


