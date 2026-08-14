# Este archivo junta embeddings de Hugging Face con un indice FAISS local. Requiere instalar los
# paquetes indicados y descarga el modelo de embeddings la primera vez. Al ejecutarlo crea el
# indice en memoria y recupera documentos para una pregunta, sin API key ni servicio externo.
# sys permite importar el corpus y los helpers desde un archivo independiente.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus que terminara dentro del indice FAISS.
from shared.dataset import DOCUMENTS
# optional_import evita errores si todavia no se instalaron los componentes locales.
from shared.utils import optional_import, show_documents

# FAISS necesita la libreria del indice y Hugging Face necesita el wrapper de LangChain.
faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS construye un VectorStore local a partir de Documents y embeddings.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings aporta los vectores que FAISS comparara.
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(DOCUMENTS, embeddings)

    # similarity_search devuelve Documents, no vectores crudos, para seguir el flujo de LangChain.
    show_documents(store.similarity_search("Como recupero mi contrasena?", k=2))
    print("FAISS mantiene el indice en memoria hasta que se guarde en disco.")

# Resumen final: FAISS combina embeddings locales y Documents para recuperar evidencia sin cloud.
# El indice en memoria es suficiente para experimentar; la persistencia aparece en un paso posterior.
