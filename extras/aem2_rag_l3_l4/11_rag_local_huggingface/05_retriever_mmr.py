# Este archivo reemplaza la busqueda pura por MMR, que equilibra similitud y diversidad. Necesita
# el mismo stack local que FAISS y Hugging Face. Al ejecutarlo se ve una seleccion menos repetida
# de Documents; modificar lambda_mult permite estudiar ese compromiso de forma directa.
# sys permite usar el corpus compartido desde un script ejecutado por separado.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS se usa para construir el indice que alimenta MMR.
from shared.dataset import DOCUMENTS
# optional_import comprueba dependencias y show_documents hace visible la diversidad devuelta.
from shared.utils import optional_import, show_documents

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS permite crear un retriever con el modo MMR de LangChain.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings calcula los vectores locales que usa el indice.
    from langchain_huggingface import HuggingFaceEmbeddings

    store = FAISS.from_documents(DOCUMENTS, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    # MMR trae mas candidatos y devuelve una mezcla de relevancia y diversidad.
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5})
    show_documents(retriever.invoke("politicas de empleados vacaciones horario remoto"))
    print("lambda_mult cerca de 1 prioriza similitud; cerca de 0 prioriza diversidad.")

# Resumen final: MMR reduce repeticion en el contexto local sin dejar de buscar relevancia.
# Ajustar lambda_mult permite decidir cuanto contexto nuevo vale frente a la similitud pura.
