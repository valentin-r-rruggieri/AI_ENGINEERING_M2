# Este archivo muestra la alternativa FAISS para busqueda vectorial totalmente local. Requiere
# instalar faiss-cpu y luego crea un indice en memoria con Documents de LangChain. Al ejecutarlo
# se ven los resultados de similitud para una pregunta de soporte; no necesita API key.
# sys permite encontrar los modulos shared desde la ejecucion directa.
import sys
# Path resuelve la ruta raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus que se almacenara en FAISS.
from shared.dataset import DOCUMENTS
# KeywordEmbeddings evita red; optional_import revisa FAISS y show_documents imprime evidencia.
from shared.utils import KeywordEmbeddings, optional_import, show_documents

# FAISS es un indice vectorial local. LangChain lo envuelve con una API muy parecida a Chroma.
if optional_import("faiss", "faiss-cpu"):
    # FAISS crea un indice vectorial local mediante la integracion comunitaria de LangChain.
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_documents(DOCUMENTS, KeywordEmbeddings())
    show_documents(store.similarity_search("Como recupero mi contrasena?", k=2))
    print("FAISS es util para aprender y prototipar busqueda vectorial en una sola maquina.")

# Resumen final: FAISS es una alternativa local para guardar y consultar vectores rapidamente.
# Su uso mediante LangChain mantiene casi la misma forma de trabajo que otros VectorStores.
