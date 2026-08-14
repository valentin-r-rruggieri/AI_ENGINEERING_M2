# Este archivo crea un VectorStore Chroma en memoria usando el corpus comun. Usa embeddings
# locales didacticos para concentrarse en la API de LangChain, sin descargar un modelo. Al
# ejecutarlo se indexan Documents y se recuperan los dos mas cercanos a una pregunta.
# sys hace disponible el paquete shared durante una ejecucion directa.
import sys
# Path localiza la carpeta del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma es el VectorStore de LangChain que indexa y busca embeddings.
from langchain_chroma import Chroma

# DOCUMENTS aporta los textos y metadata que se indexan.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings crea vectores locales y show_documents muestra los resultados.
from shared.utils import KeywordEmbeddings, show_documents

# Chroma guarda vectores junto con los Documents. KeywordEmbeddings permite
# practicar el contrato de LangChain de forma local y sin descargar un modelo.
store = Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="e09_chroma")

# similarity_search devuelve los Documents cercanos y mantiene su metadata original.
results = store.similarity_search("Puedo trabajar remoto desde casa?", k=2)
show_documents(results)
print("El VectorStore desacopla la indexacion de la consulta que hara el RAG.")

# Resumen final: Chroma une vectores y Documents para recuperar evidencia por similitud.
# El indice se puede convertir despues en un retriever y conectar con prompts o cadenas LCEL.
