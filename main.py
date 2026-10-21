import os, json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pypdf import PdfReader
from docxtpl import DocxTemplate
from docx2pdf import convert

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

def gerar_curriculo_otimizado(llm, regra_formato, modelo_pdf_texto, contexto_usuario, vaga_texto):
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
         "--- FORMATO DE SAÍDA OBRIGATÓRIO (use EXATAMENTE estes nomes de chave e tipos de bloco) ---\n"
         "{regra_formato}\n\n"
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
        "regra_formato": regra_formato,
        "modelo_pdf_texto": modelo_pdf_texto,
        "contexto_usuario": contexto_usuario,
        "vaga_texto": vaga_texto,
        # Obrigar mais ainda a resposta vir em Json para não ter erro
        "format_instructions": parser.get_format_instructions()
    })

def read_PDF(path):
    reader = PdfReader(path)
    text = ""
    for i, page in enumerate(reader.pages):
        print(f"--- Página {i + 1} ---")
        text += page.extract_text() + "\n"
    return text

def preencher_template_word(caminho_modelo, caminho_json, docx_saida, pdf_saida=None):
    # Carrega o template .docx e os dados do JSON
    doc = DocxTemplate(caminho_modelo)
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    # Injeta os dados mantendo 100% da formatação original do Word
    doc.render(dados)
    doc.save(docx_saida)
    print(f"Documento Word gerado: {docx_saida}")

    # Converte para PDF idêntico ao Word
    if pdf_saida:
        convert(docx_saida, pdf_saida)
        print(f"PDF final gerado: {pdf_saida}")

# Inicializar LLM
llm = load_llm(id_model, temperature)

# Ler o PDF
texto_pdf = read_PDF("curriculo_teste.pdf")
modelo_pdf = read_PDF("Modelo CV.pdf")

# Informações adicionais do usuário (Texto adicionais sobre a vida profissional do usuário) / texto atual apenas para testes
texto_adicional_usuario = """
Algumas coisas que não couberam no currículo:

Na Empresa Exemplo eu fiquei quase 2 anos e meio no suporte, mas o que mais gostei foi o projeto do Docker. Os servidores antigos rodavam tudo "na mão" e a gente tinha que subir cada serviço um por um. Eu fiz uns scripts em Python e Bash pra subir os containers de uma vez só, e isso ajudou bastante, o processo ficou uns 20% mais rápido segundo o que o meu coordenador comentou na época. Não tenho número exato, foi mais ou menos isso.

Também fiz por conta própria, num fim de semana que fiquei sem nada pra fazer, um bot no Telegram usando Python e SQLite. Ele responde as dúvidas que mais apareciam no suporte, tipo "como resetar a senha" ou "como configurar o e-mail no celular". O pessoal da empresa chegou a usar um tempo, mas não sei dizer quantos usuários eram. Nunca coloquei em servidor de verdade, rodava no meu PC mesmo.

Sobre SQL: nunca trabalhei com banco em empresa, mas na faculdade fiz alguns trabalhos com JOIN e GROUP BY, nada muito pesado. Já mexi um pouco com Git em projeto da faculdade e uso o GitHub pra guardar meus códigos.

No Comércio Modelo eu era estagiário e fazia de tudo um pouco: formatava computador, arrumava impressora, ajudava o pessoal quando a rede caía. Escrevi alguns passo a passo pra equipe, tipo como configurar uma impressora nova, mas nada formal.

Na faculdade já fiz dois cursos de Python e um trabalho em grupo de um sistema de controle de estoque com Java, mas não me considero bom em Java. Também fui monitor de Algoritmos por um semestre, ajudando uns colegas nas listas de exercício.

Meu inglês é bom pra ler documentação e erro no Stack Overflow, mas falar eu travo. Já participei de uma reunião com um cliente gringo uma vez, mas foi só ouvindo.

O que eu quero agora é a minha primeira vaga de dev, de preferência com Python, automação ou algo que misture com IA. Estou estudando LangChain por conta própria, mas ainda é bem básico. Não tenho experiência com cloud (AWS, Azure), então não quero que apareça como se eu tivesse.
"""

# Unir o PDF + Texto do usuário num contexto só
contexto_completo_usuario = f"{texto_pdf}\n\n--- INFORMAÇÕES ADICIONAIS DO USUÁRIO ---\n{texto_adicional_usuario}"

# Descrição da Vaga Desejada / exemplo apenas para testes
vaga_alvo = """
Vaga: Desenvolvedor(a) Python Júnior - Foco em Automação e Integrações
Empresa: Mapeia Tech

Sobre a vaga:
Estamos em busca de um(a) Desenvolvedor(a) Python Júnior para compor nosso time de tecnologia. Você será responsável por construir e manter scripts de automação de processos internos, criar integrações via APIs RESTful e auxiliar na manutenção de nossos pipelines de dados.

Requisitos necessários:
- Domínio de lógica de programação utilizando a linguagem Python.
- Conhecimento em bancos de dados relacionais (SQL) para consultas e manipulação de dados.
- Familiaridade com sistemas de controle de versão (Git/GitHub).
- Noções básicas de conteinerização com Docker.
- Boa capacidade de comunicação e raciocínio analítico para entender necessidades operacionais e transformá-las em código.

Diferenciais:
- Projetos pessoais ou experiência com consumo e criação de APIs REST.
- Vivência com automações envolvendo IA ou chatbots.
- Conhecimento básico em ambientes cloud (AWS, GCP ou Azure).
"""

# Regras da geração de formato do pdf
REGRAS_FORMATO = """
FORMATO DE SAÍDA (JSON). Você decide livremente QUAIS seções criar, a ORDEM e os TÍTULOS,
conforme o perfil do candidato e a vaga (ex.: Projetos, Voluntariado, Publicações,
Habilidades Técnicas, Certificações). Só inclua seções com conteúdo real.

Estrutura:
{
  "nome": "...", "nacionalidade": null, "estado_civil": null, "idade": null,
  "cidade_estado": "...",
  "contato": {"telefone": "...", "email": "...", "linkedin": "...", "github": null, "site": null},
  "secoes": [
    {"titulo": "Título da seção", "blocos": [ ...blocos... ]}
  ]
}

Cada bloco tem "tipo" e um destes formatos:
- {"tipo": "texto", "texto": "parágrafo corrido"}
- {"tipo": "lista", "itens": ["item 1", "item 2"]}
- {"tipo": "entrada", "titulo": "Empresa – 03/2022 a 08/2024", "subtitulo": "Cargo",
   "itens": ["responsabilidade..."], "destaques_titulo": "Principais Projetos/Resultados",
   "destaques": ["resultado..."]}
Use "entrada" para experiências, projetos, formações e cursos; "lista" para habilidades
e idiomas; "texto" para objetivo e resumo. Use apenas esses três tipos e esses nomes de chave.
"""

# Gerar Currículo Otimizado
print("Gerando currículo adaptado...")
curriculo_json = gerar_curriculo_otimizado(llm, REGRAS_FORMATO, modelo_pdf, contexto_completo_usuario, vaga_alvo)

# Salvar resultado em JSON
json_path = "curriculo_otimizado.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(curriculo_json, f, ensure_ascii=False, indent=4)

print("Currículo gerado e salvo em 'curriculo_otimizado.json' com sucesso!")

print("Gerando arquivos Word e PDF finais...")
preencher_template_word(
    caminho_modelo="Modelo CV.docx",
    caminho_json=json_path,
    docx_saida="curriculo_final.docx",
    pdf_saida="curriculo_final.pdf"
)