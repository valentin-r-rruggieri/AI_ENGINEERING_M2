# Este archivo convierte un VectorStore Chroma en un retriever de LangChain. El retriever recibe
# una pregunta y devuelve Documents relevantes, que luego podrian alimentar un prompt. Al
# ejecutarlo se ve la lista de evidencia recuperada para una consulta de soporte.
# sys permite que el script encuentre shared al correrlo desde cualquier carpeta.
import sys
# Path calcula la raiz del curso para ese import.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma guarda vectores y expone as_retriever para recuperar Documents.
from langchain_chroma import Chroma

# DOCUMENTS es el corpus que se cargara en el VectorStore.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings crea vectores locales y show_documents muestra los Documents recuperados.
from shared.utils import KeywordEmbeddings, show_documents

# as_retriever convierte el VectorStore en la pieza que recibe una pregunta y devuelve Documents.
retriever = Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="e12").as_retriever(search_kwargs={"k": 2})

# invoke es la interfaz comun de los Runnables de LangChain.
show_documents(retriever.invoke("Necesito recuperar mi contrasena"))
print("El retriever marca el limite entre la pregunta del usuario y la evidencia del prompt.")

# Resumen final: un retriever recibe una consulta y devuelve Documents utiles para responder.
# Esta separacion permite cambiar el indice sin reescribir el prompt ni la parte generativa.
