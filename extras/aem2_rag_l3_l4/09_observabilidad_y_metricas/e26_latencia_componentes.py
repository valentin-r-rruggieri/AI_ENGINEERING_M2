# Este archivo mide por separado splitting, embeddings y retrieval para mostrar que la latencia
# total de un RAG tiene varias causas. Usa implementaciones locales para que se pueda ejecutar
# sin clave. Al terminar imprime milisegundos por etapa y facilita decidir que optimizar.
# sys habilita los imports del curso desde este archivo individual.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# RecursiveCharacterTextSplitter es la etapa de chunking que se medira.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# DOCUMENTS aporta la entrada del pipeline medido.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings, elapsed y lexical_search representan embeddings, cronometro y retrieval.
from shared.utils import KeywordEmbeddings, elapsed, lexical_search

# Medir etapas por separado muestra donde se gasta el tiempo de un pipeline RAG.
with elapsed("Splitting"):
    chunks = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10).split_documents(DOCUMENTS)

with elapsed("Embeddings"):
    KeywordEmbeddings().embed_documents([chunk.page_content for chunk in chunks])

with elapsed("Retrieval"):
    lexical_search("vacaciones", chunks, k=2)

print("La latencia total no sirve para optimizar si no se conoce el costo de cada componente.")

# Resumen final: medir cada etapa revela si el problema esta en splitting, embeddings o retrieval.
# Las optimizaciones deben elegirse con estas mediciones y no solo por la sensacion de lentitud.
