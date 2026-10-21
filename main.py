import os, json
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

def gerar_curriculo_otimizado(llm, modelo_pdf_texto, contexto_usuario, vaga_texto):
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
         "--- PDF MODELO DE REFERÊNCIA (Siga este layout/estrutura) ---\n"
         "{modelo_pdf_texto}\n\n"
         "--- HISTÓRICO E CURRÍCULO DO CANDIDATO ---\n"
         "{contexto_usuario}\n\n"
         "--- DESCRIÇÃO DA VAGA ALVO ---\n"
         "{vaga_texto}\n\n"
         "Gere o currículo otimizado no formato JSON."
        )
    ])

    chain = template | llm | parser
    return chain.invoke({
        "modelo_pdf_texto": modelo_pdf_texto,
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

# Ler o PDF
texto_pdf = read_PDF("curriculo_teste.pdf")
modelo_pdf = read_PDF("Modelo CV.pdf")

# Informações adicionais do usuário (Texto adicionais sobre a vida profissional do usuário) / texto atual apenas para testes
texto_adicional_usuario = """
Conquistas e detalhes adicionais:
- No projeto de migração para Docker na Empresa Exemplo Ltda., criei scripts em Python e Bash para automatizar a subida dos containers, reduzindo o tempo do processo em 20%.
- Desenvolvi por conta própria uma automação em Python usando a API do Telegram e SQLite para responder dúvidas frequentes dos usuários da empresa.
- Tenho experiência prática em consultas SQL intermediárias (JOINs, GROUP BY) desenvolvidas em projetos da faculdade de Engenharia da Computação.
- Tenho interesse em atuar com Python, automação de processos, APIs e integração com modelos de IA.
"""

# Unir o PDF + Texto do usuário num contexto só
contexto_completo_usuario = f"{texto_pdf}\n\n--- INFORMAÇÕES ADICIONAIS DO USUÁRIO ---\n{texto_adicional_usuario}"

# Descrição da Vaga Desejada / exemplo apenas para testes
vaga_alvo = """
Vaga: Desenvolvedor Python Júnior / Automação
Requisitos:
- Conhecimento em linguagem Python para automação de tarefas e criação de scripts.
- Domínio de banco de dados SQL e controle de versão com Git.
- Vivência com Linux e ambientes conteinerizados (Docker).
- Boa comunicação para suporte e entendimento de chamados/requisitos.
- Diferencial: Conhecimento básico em APIs, bots ou integração com IA.
"""

# Gerar Currículo Otimizado
print("Gerando currículo adaptado...")
curriculo_json = gerar_curriculo_otimizado(llm, contexto_completo_usuario, vaga_alvo)

# Salvar resultado em ficheiro JSON
with open("curriculo_otimizado.json", "w", encoding="utf-8") as f:
    json.dump(curriculo_json, f, ensure_ascii=False, indent=4)

print("Currículo gerado e salvo em 'curriculo_otimizado.json' com sucesso!")
