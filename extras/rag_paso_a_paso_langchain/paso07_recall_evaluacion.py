"""Paso 7: medir si FAISS sigue encontrando lo mismo que el InMemoryVectorStore (Recall@K).

Mismo mecanismo que en el paso a paso manual, pero comparando los dos vector
stores de LangChain en vez de comparar a mano.
"""
from comun import PREGUNTA
from paso03_vectorstore_memoria import buscar_exacto
from paso05_vectorstore_faiss import buscar_faiss

PREGUNTAS_DE_PRUEBA = [
    PREGUNTA,
    "Como descargo mi factura?",
    "Me llego un cobro dos veces, que hago?",
    "Como recupero mi contrasena?",
]


def recall_at_k(pregunta: str, k: int = 3) -> float:
    exacto = {documento.metadata["chunk_id"] for documento, _ in buscar_exacto(pregunta, top_k=k)}
    faiss_resultado = {documento.metadata["chunk_id"] for documento, _ in buscar_faiss(pregunta, top_k=k)}
    return len(exacto & faiss_resultado) / len(exacto)


if __name__ == "__main__":
    recalls = []
    for pregunta in PREGUNTAS_DE_PRUEBA:
        recall = recall_at_k(pregunta)
        recalls.append(recall)
        print(f"Recall@3 para '{pregunta}': {recall:.2f}")
    print(f"\nRecall@3 promedio: {sum(recalls) / len(recalls):.2f}")
