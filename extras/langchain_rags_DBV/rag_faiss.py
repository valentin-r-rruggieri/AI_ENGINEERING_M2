"""RAG simple con LangChain y FAISS.

Documentación: https://docs.langchain.com/oss/python/integrations/vectorstores/faiss
"""

# %% 1. Configuración
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
QUESTION = "¿Cuántos días de vacaciones tengo?"
TOP_K = 3

# -----------------------------------------------------------------------------
# MODELOS: cambiar solo estas variables en .env, antes de ejecutar el notebook.
# -----------------------------------------------------------------------------
# "openai" usa OpenAI. "local" usa LM Studio. "openrouter" usa OpenRouter.
MODEL_MODE = os.getenv("MODEL_MODE", "openai")

# El LLM redacta la respuesta; el modelo de embeddings transforma texto en vectores.
if MODEL_MODE == "local":
    CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL")
    EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL")
    API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
elif MODEL_MODE == "openrouter":
    CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL")
    EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL")
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    BASE_URL = "https://openrouter.ai/api/v1"
else:
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = None

# Son wrappers de LangChain; el resto del RAG no cambia al alternar entre cloud y local.
# https://python.langchain.com/docs/integrations/text_embedding/openai/
# LM Studio espera strings; desactivamos la tokenización previa solo en modo local.
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL, tiktoken_enabled=MODEL_MODE != "local", check_embedding_ctx_length=MODEL_MODE != "local")
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=API_KEY, base_url=BASE_URL)

# %% 2. Cargar Documents con LangChain
# DirectoryLoader carga todos los Markdown y .load() devuelve Documents de LangChain.
# https://python.langchain.com/docs/integrations/document_loaders/directory/
documents = DirectoryLoader(str(BASE_DIR / "data"), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()

# queries.md es para evaluación, por eso no forma parte de la base de conocimiento.
documents = [doc for doc in documents if Path(doc.metadata["source"]).name != "queries.md"]

# La metadata permite saber después de qué fuente y departamento salió cada respuesta.
for doc in documents:
    doc.metadata["source"] = Path(doc.metadata["source"]).name
    doc.metadata["department"] = re.search(r"department:\s*([^\n]+)", doc.page_content).group(1)
    doc.page_content = doc.page_content.split("---", 2)[2]

# %% 3. Chunks y base vectorial
# Splitter de LangChain: https://python.langchain.com/docs/concepts/text_splitters/
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

# FAISS guarda los embeddings en memoria: al reiniciar, el índice se vuelve a crear.
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

# %% 4. RAG
retrieved = retriever.invoke(QUESTION)

# Los chunks recuperados forman el único contexto que recibirá el LLM.
context = "\n\n".join(f"Fuente: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved)

prompt = ChatPromptTemplate.from_template("Responde solo usando el contexto. Cita las fuentes.\n\nContexto:\n{context}\n\nPregunta: {question}")

answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": QUESTION})

print("Fuentes:", [doc.metadata["source"] for doc in retrieved])
print("\nRespuesta:\n", answer)
