"""RAG simple con LangChain y Pinecone.

Documentación: https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone
"""

# %% 1. Configuración
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

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

# El LLM y el embedding model se pueden cambiar sin modificar la parte vectorial.
# El índice "henry" tiene dimensión 512; el embedding debe devolver esa misma dimensión.
PINECONE_EMBEDDING_DIMENSIONS = int(os.getenv("PINECONE_EMBEDDING_DIMENSIONS", "0")) or None
# LM Studio espera strings; desactivamos la tokenización previa solo en modo local.
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL, dimensions=PINECONE_EMBEDDING_DIMENSIONS, tiktoken_enabled=MODEL_MODE != "local", check_embedding_ctx_length=MODEL_MODE != "local")
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, api_key=API_KEY, base_url=BASE_URL)

# %% 2. Cargar Documents con LangChain
# https://python.langchain.com/docs/integrations/document_loaders/directory/
documents = DirectoryLoader(str(BASE_DIR / "data"), glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
documents = [doc for doc in documents if Path(doc.metadata["source"]).name != "queries.md"]

# La fuente y el departamento permanecen asociados a cada chunk dentro de Pinecone.
for doc in documents:
    doc.metadata["source"] = Path(doc.metadata["source"]).name
    doc.metadata["department"] = re.search(r"department:\s*([^\n]+)", doc.page_content).group(1)
    doc.page_content = doc.page_content.split("---", 2)[2]

# %% 3. Chunks y base vectorial
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

# Conectamos al índice cloud que ya creaste en Pinecone.
# El nombre se define en .env: PINECONE_INDEX_NAME=quickstart
pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]
namespace = os.getenv("PINECONE_NAMESPACE", "clase-rag")
index = pinecone.Index(index_name)

# La dimensión del índice debe coincidir con la dimensión del embedding model elegido.
index_dimension = pinecone.describe_index(index_name).dimension
embedding_dimension = len(embeddings.embed_query("comprobar dimensión"))
if index_dimension != embedding_dimension:
    raise ValueError(f"El índice usa {index_dimension} dimensiones y el modelo genera {embedding_dimension}. Elegí un modelo compatible o creá un índice con esa dimensión.")

# IDs estables: si volvés a ejecutar, los mismos chunks se actualizan en vez de duplicarse.
chunk_ids = [f"chunk-{number}" for number in range(1, len(chunks) + 1)]
print(f"Conectado a Pinecone Cloud: índice={index_name}, namespace={namespace}")
print(f"Subiendo {len(chunks)} chunks:")
for chunk_id, chunk in zip(chunk_ids, chunks):
    print(f"\n{chunk_id} | fuente={chunk.metadata['source']} | departamento={chunk.metadata['department']}")
    print(chunk.page_content.strip())

vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
vector_store.add_documents(chunks, ids=chunk_ids)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
print("\nChunks cargados. Verlos en Pinecone: índice → namespace clase-rag.")

# %% 4. RAG
retrieved = retriever.invoke(QUESTION)

context = "\n\n".join(f"Fuente: {doc.metadata['source']}\n{doc.page_content}" for doc in retrieved)

prompt = ChatPromptTemplate.from_template("Responde solo usando el contexto. Cita las fuentes.\n\nContexto:\n{context}\n\nPregunta: {question}")

answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": QUESTION})

print("Fuentes:", [doc.metadata["source"] for doc in retrieved])
print("\nRespuesta:\n", answer)
