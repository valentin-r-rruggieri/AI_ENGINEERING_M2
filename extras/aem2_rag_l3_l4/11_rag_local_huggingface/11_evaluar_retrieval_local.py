# Este archivo mide si el RAG local recupera la evidencia correcta antes de juzgar respuestas de
# un LLM. Construye FAISS con embeddings Hugging Face y compara resultados con consultas ya
# etiquetadas. Al ejecutarlo se imprime Recall@3 por pregunta y un promedio para el modelo local.
# sys permite usar el dataset de evaluacion desde este script independiente.
import sys
# Path calcula la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus y EVALUATION_QUERIES contiene los IDs relevantes esperados.
from shared.dataset import DOCUMENTS, EVALUATION_QUERIES
# optional_import comprueba el stack local antes de medir retrieval real.
from shared.utils import optional_import

faiss_ready = optional_import("faiss", "faiss-cpu")
hf_ready = optional_import("langchain_huggingface", "langchain-huggingface sentence-transformers")

if faiss_ready and hf_ready:
    # FAISS recupera los IDs que se compararan contra las etiquetas de evaluacion.
    from langchain_community.vectorstores import FAISS
    # HuggingFaceEmbeddings define el espacio vectorial local evaluado.
    from langchain_huggingface import HuggingFaceEmbeddings

    store = FAISS.from_documents(DOCUMENTS, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    recalls = []
    for item in EVALUATION_QUERIES:
        ids = {doc.metadata["id"] for doc in store.similarity_search(item["query"], k=3)}
        recall = len(ids & item["relevant_ids"]) / len(item["relevant_ids"])
        recalls.append(recall)
        print(f"{item['query']} -> recall@3={recall:.2f} ids={sorted(ids)}")

    print(f"\nRecall@3 promedio: {sum(recalls) / len(recalls):.2f}")
    print("La eleccion del modelo local debe basarse en estas metricas, no solo en que cargue rapido.")

# Resumen final: evaluar FAISS y embeddings locales con casos etiquetados evita elegir por intuicion.
# Recall@3 indica si la evidencia correcta aparece entre los documentos que el RAG podra usar.
