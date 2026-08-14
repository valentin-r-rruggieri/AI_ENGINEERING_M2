# Este archivo separa dos controles de calidad del retrieval: Top-K define cuantos resultados
# se consideran y threshold descarta los que son demasiado debiles. Al ejecutarlo se imprimen
# los scores de los candidatos y se indica cuales llegan al contexto del RAG.
# sys habilita los imports compartidos en la ejecucion directa.
import sys
# Path ubica la raiz del curso desde este archivo.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Chroma permite pedir resultados junto con su distancia vectorial.
from langchain_chroma import Chroma

# DOCUMENTS es la base de evidencia para la busqueda.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings produce vectores deterministas sin usar una API externa.
from shared.utils import KeywordEmbeddings

# Top-K limita cuantos candidatos llegan al contexto; no garantiza que todos sean buenos.
store = Chroma.from_documents(DOCUMENTS, KeywordEmbeddings(), collection_name="e13")
threshold = 0.35

# Chroma entrega distancia: menor es mejor. Este score didactico permite aplicar un umbral visible.
for document, distance in store.similarity_search_with_score("Cuantos dias de vacaciones tengo?", k=4):
    score = 1 / (1 + distance)
    accepted = score >= threshold
    print(f"id={document.metadata['id']:<11} score={score:.3f} {'ACEPTADO' if accepted else 'DESCARTADO'}")

print("El threshold ayuda a abstenerse cuando no hay evidencia con similitud suficiente.")

# Resumen final: Top-K controla cantidad y threshold controla calidad minima de la evidencia.
# Usarlos juntos reduce el riesgo de que un LLM responda apoyado en documentos irrelevantes.
