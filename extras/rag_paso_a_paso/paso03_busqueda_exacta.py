"""Paso 3: buscar a mano, comparando la pregunta contra todos los chunks (k-NN exacto).

Es la busqueda mas simple posible. Sirve como referencia de calidad para
los pasos siguientes (FAISS, filtros, evaluacion).
"""
import json

import numpy as np

from comun import ESTADO, PREGUNTA, embed

with open(ESTADO / "02_chunks_con_embeddings.json", encoding="utf-8") as f:
    chunks = json.load(f)


def buscar_exacto(pregunta: str, top_k: int = 3) -> list[dict]:
    """Compara la pregunta contra TODOS los chunks y devuelve los top_k mas parecidos."""
    vector_pregunta = embed(pregunta)
    resultados = []
    for chunk in chunks:
        vector_chunk = np.array(chunk["embedding"])
        score = np.dot(vector_pregunta, vector_chunk) / (
            np.linalg.norm(vector_pregunta) * np.linalg.norm(vector_chunk)
        )
        resultados.append({**chunk, "score": float(score)})
    return sorted(resultados, key=lambda r: r["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    print(f"Pregunta: {PREGUNTA}")
    for resultado in buscar_exacto(PREGUNTA):
        print(f"  [{resultado['score']:.4f}] ({resultado['metadata']['source']}) {resultado['text']}")
