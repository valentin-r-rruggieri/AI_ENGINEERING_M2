# Este archivo presenta el equivalente local del RAG cloud: Hugging Face genera embeddings y
# FAISS busca los documentos, todo dentro de la maquina. Requiere paquetes opcionales y descarga
# inicial del modelo. Al ejecutarlo se obtiene evidencia local lista para conectar a un LLM.
# sys agrega los modulos compartidos al path de ejecucion.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus que se indexa en la alternativa local.
from shared.dataset import DOCUMENTS
# optional_import valida paquetes opcionales y show_documents imprime la evidencia.
from shared.utils import optional_import, show_documents

# Esta ruta local necesita un indice FAISS y un modelo sentence-transformers instalados.
faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS implementa el indice vectorial local dentro de LangChain.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings conecta un modelo sentence-transformers como proveedor local.
    from langchain_huggingface import HuggingFaceEmbeddings

    # La construccion del store es igual a la cloud; solo cambia el proveedor de embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(DOCUMENTS, embeddings)
    show_documents(store.similarity_search("Puedo trabajar desde casa?", k=2))
    print("La recuperacion local puede conectarse despues a un modelo generativo local.")

# Resumen final: un RAG local puede recuperar evidencia sin depender de APIs ni servicios cloud.
# FAISS y Hugging Face respetan los contratos de LangChain y facilitan cambiar de arquitectura.
