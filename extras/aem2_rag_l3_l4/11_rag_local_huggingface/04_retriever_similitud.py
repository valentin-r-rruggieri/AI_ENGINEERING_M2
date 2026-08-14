# Este archivo convierte el indice FAISS local en un retriever de similitud de LangChain. El
# resultado ya no son vectores sino Documents con texto y metadata, listos para ser contexto.
# Al ejecutarlo se recuperan los dos documentos mas cercanos a una pregunta sobre horario.
# sys permite cargar modulos del curso desde una ejecucion directa.
import sys
# Path ubica la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es la evidencia que se indexara para este ejemplo.
from shared.dataset import DOCUMENTS
# optional_import verifica componentes locales y show_documents imprime los resultados.
from shared.utils import optional_import, show_documents

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS implementa el VectorStore local que se convertira en retriever.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings genera vectores locales para corpus y pregunta.
    from langchain_huggingface import HuggingFaceEmbeddings

    store = FAISS.from_documents(DOCUMENTS, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    # as_retriever devuelve un Runnable que acepta una pregunta a traves de invoke.
    retriever = store.as_retriever(search_kwargs={"k": 2})
    show_documents(retriever.invoke("Cual es el horario laboral?"))
    print("El retriever separa la recuperacion de la futura etapa de generacion.")

# Resumen final: as_retriever convierte FAISS en una pieza reutilizable dentro de cadenas LangChain.
# La similitud recupera los documentos mas cercanos, pero todavia no garantiza diversidad ni grounding.
