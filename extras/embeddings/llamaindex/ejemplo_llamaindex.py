"""El mismo RAG que en manual/, pero con LlamaIndex haciendo el trabajo pesado."""
from pathlib import Path
import sys

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from documentos import DOCUMENTOS, PREGUNTA  # noqa: E402

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 120
CHUNK_OVERLAP = 20
TOP_K = 3

# 0. Configuramos que modelos y que "cortador" de texto va a usar LlamaIndex.
# OpenAIEmbedding docs: https://docs.llamaindex.ai/en/stable/examples/embeddings/OpenAI/
# Settings docs: https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/
Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
Settings.llm = OpenAI(model=CHAT_MODEL, temperature=0)
# SentenceSplitter docs: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencesplitter
Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# 1. Chunking: cada documento se convierte en "nodos" (los chunks de LlamaIndex).
# Document docs: https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
docs = [Document(text=doc["text"], metadata={"source": doc["source"]}) for doc in DOCUMENTOS]
nodes = Settings.node_parser.get_nodes_from_documents(docs)
print(f"Nodos generados: {len(nodes)}")

# 2. Embeddings + indice: LlamaIndex calcula los vectores y arma el indice vectorial.
# VectorStoreIndex docs: https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
index = VectorStoreIndex(nodes)

# 3. Busqueda semantica: traemos los nodos mas parecidos a la pregunta.
# Retriever docs: https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/
retriever = index.as_retriever(similarity_top_k=TOP_K)
top_3 = retriever.retrieve(PREGUNTA)

print("\nTop 3 nodos mas parecidos a la pregunta:")
for item in top_3:
    print(f"  [{item.score:.4f}] ({item.node.metadata['source']}) {item.node.get_content()}")

# 4. RAG: el motor de consultas recupera contexto y genera la respuesta en un solo paso.
# Query engine docs: https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
query_engine = index.as_query_engine(similarity_top_k=TOP_K)
respuesta = query_engine.query(PREGUNTA)

print(f"\nPregunta: {PREGUNTA}")
print(f"Respuesta: {respuesta}")
