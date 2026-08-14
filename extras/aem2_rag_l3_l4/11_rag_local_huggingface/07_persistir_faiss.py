# Este archivo ensena a guardar y reabrir un indice FAISS. Requiere Hugging Face y FAISS, crea
# una carpeta dentro de storage y luego carga desde ella para buscar sin reconstruir el indice.
# Al ejecutarlo se imprime la ruta persistida y un resultado recuperado despues de la reapertura.
# shutil elimina el indice creado por una ejecucion anterior para que el ejemplo sea repetible.
import shutil
# sys permite acceder al corpus y helpers compartidos.
import sys
# Path construye la ruta local donde se guardara el indice.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus que se guardara dentro de FAISS.
from shared.dataset import DOCUMENTS
# ROOT entrega la carpeta del curso y optional_import valida paquetes locales.
from shared.utils import ROOT, optional_import, show_documents

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS permite crear y reabrir un indice persistido con LangChain.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings debe ser el mismo modelo usado al guardar y al consultar.
    from langchain_huggingface import HuggingFaceEmbeddings

    directory = ROOT / "storage" / "huggingface_faiss"
    if directory.exists():
        shutil.rmtree(directory)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    FAISS.from_documents(DOCUMENTS, embeddings).save_local(str(directory))
    # allow_dangerous_deserialization es requerido por FAISS al restaurar su docstore local.
    reloaded = FAISS.load_local(str(directory), embeddings, allow_dangerous_deserialization=True)
    print("Indice guardado en:", directory)
    show_documents(reloaded.similarity_search("cobertura medica", k=1))
    print("Persistir evita reindexar, pero requiere controlar corpus y version del modelo.")

# Resumen final: guardar FAISS permite reiniciar una aplicacion sin recalcular todos los vectores.
# El modelo de embeddings y la version del corpus deben mantenerse compatibles con el indice guardado.
