# Este archivo muestra como reducir el contexto de un RAG local mediante metadata. FAISS primero
# recupera candidatos y despues el codigo conserva solo la categoria autorizada. Al ejecutarlo
# se ve que una pregunta de RRHH no mezcla documentos de soporte, producto o beneficios.
# sys permite importar los datos compartidos desde la carpeta del curso.
import sys
# Path encuentra la raiz del curso desde este archivo.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS tiene categorias que se pueden usar para acotar la busqueda local.
from shared.dataset import DOCUMENTS
# optional_import comprueba las dos dependencias necesarias y show_documents imprime evidencia.
from shared.utils import optional_import, show_documents

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS almacena los vectores y metadata de los Documents locales.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings vectoriza el corpus sin hacer llamadas cloud.
    from langchain_huggingface import HuggingFaceEmbeddings

    store = FAISS.from_documents(DOCUMENTS, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    # FAISS no filtra metadata en el indice; fetch_k trae candidatos para filtrarlos localmente.
    candidates = store.similarity_search("Cual es el horario?", k=6)
    rrhh_only = [document for document in candidates if document.metadata["category"] == "rrhh"]
    show_documents(rrhh_only)
    print("El filtro limita la evidencia antes de que llegue al prompt del modelo local.")

# Resumen final: filtrar metadata antes de generar evita mezclar dominios o informacion no autorizada.
# En indices locales simples el filtro puede hacerse despues de recuperar candidatos suficientes.
