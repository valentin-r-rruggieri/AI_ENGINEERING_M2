"""El mismo RAG que en manual/, pero con LangChain haciendo el trabajo pesado."""
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from documentos import DOCUMENTOS, PREGUNTA  # noqa: E402

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

# 1. Chunking: LangChain parte los textos respetando el tamano configurado.
# Docs: https://python.langchain.com/docs/how_to/recursive_text_splitter/
# API ref: https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
docs = splitter.create_documents(
    texts=[doc["text"] for doc in DOCUMENTOS],
    metadatas=[{"source": doc["source"]} for doc in DOCUMENTOS],
)
print(f"Chunks generados: {len(docs)}")

# 2. Embeddings + vector store: LangChain calcula los vectores y los indexa en memoria.
# OpenAIEmbeddings docs: https://python.langchain.com/docs/integrations/text_embedding/openai/
# InMemoryVectorStore docs: https://python.langchain.com/docs/integrations/vectorstores/in_memory/
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = InMemoryVectorStore.from_documents(docs, embedding_model)

# 3. Busqueda semantica: traemos los 3 chunks mas parecidos a la pregunta.
# Docs: https://python.langchain.com/docs/concepts/vectorstores/#similarity-search
top_3 = vectorstore.similarity_search_with_score(PREGUNTA, k=3, score_threshold=0.6 ,filter=[{"source": "documento_1.txt"}])

print("\nTop 3 chunks mas parecidos a la pregunta:")
for doc, score in top_3:
    print(f"  [{score:.4f}] ({doc.metadata['source']}) {doc.page_content}")

# 4. RAG: armamos el prompt con el contexto recuperado y pedimos la respuesta.
# ChatPromptTemplate docs: https://python.langchain.com/docs/concepts/prompt_templates/
contexto = "\n".join(f"({doc.metadata['source']}) {doc.page_content}" for doc, _ in top_3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Responde solo usando el contexto."),
        ("human", "Pregunta: {pregunta}\n\nContexto:\n{contexto}"),
    ]
)
mensajes = prompt.invoke({"pregunta": PREGUNTA, "contexto": contexto})

# ChatOpenAI docs: https://python.langchain.com/docs/integrations/chat/openai/
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
respuesta = llm.invoke(mensajes)

print(f"\nPregunta: {PREGUNTA}")
print(f"Respuesta: {respuesta.content}")
