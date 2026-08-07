"""Paso 5: reemplazar la busqueda lineal del paso 3 por un indice FAISS.

Mismo resultado que paso03_busqueda_exacta.py, pero con una libreria pensada
para escalar a millones de vectores, y persistiendo el indice a disco.
Docs: https://faiss.ai/ y https://github.com/facebookresearch/faiss/wiki
"""
import json

import faiss
import numpy as np

from comun import ESTADO, PREGUNTA, embed, normalizar

with open(ESTADO / "02_chunks_con_embeddings.json", encoding="utf-8") as f:
    chunks = json.load(f)

# 1. Normalizamos los vectores: asi el producto interno de FAISS se comporta como coseno.
vectores = normalizar(np.array([chunk["embedding"] for chunk in chunks])).astype("float32")

# 2. Creamos el indice y agregamos los vectores. FAISS solo conoce posiciones, no chunk_id,
# por eso guardamos aparte que posicion corresponde a que chunk.
dimension = vectores.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(vectores)
posicion_a_chunk = {posicion: chunk["chunk_id"] for posicion, chunk in enumerate(chunks)}

faiss.write_index(index, str(ESTADO / "05_indice.faiss"))
with open(ESTADO / "05_posicion_a_chunk.json", "w", encoding="utf-8") as f:
    json.dump(posicion_a_chunk, f)

print(f"Indice FAISS creado con {index.ntotal} vectores, guardado en {ESTADO / '05_indice.faiss'}")


def buscar_faiss(pregunta: str, top_k: int = 3) -> list[dict]:
    """Busca los top_k chunks mas parecidos usando el indice FAISS."""
    vector_pregunta = normalizar(embed(pregunta).reshape(1, -1)).astype("float32")
    scores, posiciones = index.search(vector_pregunta, top_k)
    resultados = []
    for score, posicion in zip(scores[0], posiciones[0]):
        chunk_id = posicion_a_chunk[int(posicion)]
        chunk = next(c for c in chunks if c["chunk_id"] == chunk_id)
        resultados.append({**chunk, "score": float(score)})
    return resultados


if __name__ == "__main__":
    print(f"\nPregunta: {PREGUNTA}")
    for resultado in buscar_faiss(PREGUNTA):
        print(f"  [{resultado['score']:.4f}] ({resultado['metadata']['source']}) {resultado['text']}")
