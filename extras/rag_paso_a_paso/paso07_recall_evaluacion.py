"""Paso 7: medir si FAISS sigue encontrando lo mismo que la busqueda exacta (Recall@K).

Comparamos, para varias preguntas, el Top-K de la busqueda exacta (paso 3)
contra el Top-K de FAISS (paso 5). Con un corpus tan chico deberian coincidir
siempre, pero este es el mismo mecanismo que se usa para validar un indice
aproximado (HNSW, IVF) sobre un corpus real y grande.
"""
from comun import PREGUNTA
from paso03_busqueda_exacta import buscar_exacto
from paso05_indice_faiss import buscar_faiss

PREGUNTAS_DE_PRUEBA = [
    PREGUNTA,
    "Como descargo mi factura?",
    "Me llego un cobro dos veces, que hago?",
    "Como recupero mi contrasena?",
]


def recall_at_k(pregunta: str, k: int = 3) -> float:
    exacto = {r["chunk_id"] for r in buscar_exacto(pregunta, top_k=k)}
    faiss_resultado = {r["chunk_id"] for r in buscar_faiss(pregunta, top_k=k)}
    return len(exacto & faiss_resultado) / len(exacto)


if __name__ == "__main__":
    recalls = []
    for pregunta in PREGUNTAS_DE_PRUEBA:
        recall = recall_at_k(pregunta)
        recalls.append(recall)
        print(f"Recall@3 para '{pregunta}': {recall:.2f}")
    print(f"\nRecall@3 promedio: {sum(recalls) / len(recalls):.2f}")
