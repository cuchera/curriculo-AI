import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

def gerar_curriculo_otimizado(llm, contexto_usuario, vaga_texto):
    # Objeto para obrigar a resposta vir em json 
    parser = JsonOutputParser()

    template = ChatPromptTemplate.from_messages([
        ("system", 
         "Você é um especialista em recrutamento e otimização de currículos para ATS.\n\n"
         "OBJETIVO:\n"
         "Reescrever e organizar as experiências do candidato para que fiquem o mais aderentes "
         "possível à vaga desejada, destacando palavras-chave e conquistas relevantes.\n\n"
         "REGRAS INEGOCIÁVEIS:\n"
         "1. NUNCA invente ou infira habilidades, tecnologias, cargos, datas ou resultados que não "
         "estejam explicitamente mencionados no histórico do candidato.\n"
         "2. Se a vaga exige um requisito que o candidato NÃO possui no histórico, simplesmente OMITA "
         "ou mantenha a informação real existente. Não tente inventar para preencher a lacuna.\n"
         "3. Reorganize e reescreva tópicos para enfatizar pontos fortes que coincidem com a vaga.\n"
         "4. A resposta deve ser estritamente no formato JSON estruturado.\n"
        ),
        ("human",
         "--- HISTÓRICO E CURRÍCULO DO CANDIDATO ---\n"
         "{contexto_usuario}\n\n"
         "--- DESCRIÇÃO DA VAGA ALVO ---\n"
         "{vaga_texto}\n\n"
         "Gere o currículo otimizado no formato JSON."
        )
    ])

    chain = template | llm | parser
    return chain.invoke({
        "contexto_usuario": contexto_usuario,
        "vaga_texto": vaga_texto,
        # Obrigar mais ainda a resposta vir em Json para não ter erro
        "format_instruction": parser.get_format_instructions()
    })

def read_PDF(path):
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        print(f"--- Página {i + 1} ---")
        text += page.extract_text() + "\n"
    return text

# Inicializar LLM
llm = load_llm(id_model, temperature)

curriculo = read_PDF("curriculo_teste.pdf")
print(curriculo[0:500])