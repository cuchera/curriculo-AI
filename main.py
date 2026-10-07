import os, requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()
api_key = os.getenv("ChatGroq_API_KEY")

id_model = "openai/gpt-oss-120b"
temperature = 0

def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    api_key=api_key,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )
  return llm

def llm_generate(llm, prompt):
    template = ChatPromptTemplate.from_messages([
        ("system", 
         "Você é um recrutador experiente no Brasil.\n"
         "Regras:\n"
         "- Use apenas informações presentes no currículo. Nunca invente experiências, ferramentas ou resultados.\n"
         "- Se um requisito da vaga não aparece no currículo, marque como não encontrado.\n"
         "- Não exagere o nível de conhecimento do candidato.\n"
         "- Responda apenas com JSON, sem texto antes ou depois.\n"
         "- Pode responder duvidas do usuario também"),
        ("human",
         "{prompt}")
    ])

    chain = template | llm | StrOutputParser()
    res = chain.invoke({"prompt": prompt})
    return res

def read_PDF(path):
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        print(f"--- Página {i + 1} ---")
        print(page.extract_text())
        text += page.extract_text() + "\n"
    return text


llm = load_llm(id_model, temperature)

curriculo = read_PDF("curriculo_teste.pdf")
print(curriculo[0:500])