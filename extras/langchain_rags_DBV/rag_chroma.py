"""RAG simple con LangChain y Chroma.

Documentación: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
"""

# %% 1. Configuración
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
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
MODEL_MODE = os.getenv("MODEL_MODE", "openai")  # "openai", "local" o "openrouter".
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

# El LLM genera texto y el embedding model permite buscar significado.
# LM Studio espera strings; desactivamos la tokenización previa solo en modo local.
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL, tiktoken_enabled=MODEL_MODE != "local", check_embedding_ctx_length=MODEL_MODE != "local")
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=API_KEY, base_url=BASE_URL)

# %% 2. Cargar Documents con LangChain
# https://python.langchain.com/docs/integrations/document_loaders/directory/
documents = DirectoryLoader(str(BASE_DIR / "data"), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
documents = [doc for doc in documents if Path(doc.metadata["source"]).name != "queries.md"]

# Conservamos fuente y departamento dentro de cada Document para trazabilidad.
for doc in documents:
    doc.metadata["source"] = Path(doc.metadata["source"]).name
    doc.metadata["department"] = re.search(r"department:\s*([^\n]+)", doc.page_content).group(1)
    doc.page_content = doc.page_content.split("---", 2)[2]

# %% 3. Chunks y base vectorial
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]).split_documents(documents)

# Chroma persiste localmente. Reiniciamos la colección para que el notebook no duplique documentos.
vector_store = Chroma(collection_name="aem2_langchain", embedding_function=embeddings, persist_directory=str(BASE_DIR / "storage" / "chroma"))
vector_store.reset_collection()
vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K, "score_threshold": 0.5})

# %% 4. RAG
retrieved = retriever.invoke(QUESTION)

context = "\n\n".join(f"Fuente: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved)

prompt = ChatPromptTemplate.from_template("Responde solo usando el contexto. Cita las fuentes.\n\nContexto:\n{context}\n\nPregunta: {question}")

answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": QUESTION})

print("Fuentes:", [doc.metadata["source"] for doc in retrieved])
print("\nRespuesta:\n", answer)
