"""Paso 4: por que normalizar antes de usar producto interno (la "trampa de la magnitud").

La funcion normalizar() que vamos a usar desde el paso 5 en adelante vive en
comun.py; aca la explicamos y la probamos.
"""
import json

import numpy as np

from comun import ESTADO, normalizar

# 1. Ejemplo de juguete en 2D: el caso clasico donde el producto interno sin normalizar miente.
chunk_largo = np.array([10.0, 1.0])
chunk_corto = np.array([2.0, 2.0])
pregunta = np.array([1.0, 1.0])

print("Ejemplo de juguete (sin normalizar):")
for nombre, chunk in [("chunk largo", chunk_largo), ("chunk corto", chunk_corto)]:
    producto_interno = chunk @ pregunta
    coseno = producto_interno / (np.linalg.norm(chunk) * np.linalg.norm(pregunta))
    print(f"  {nombre}: producto_interno={producto_interno:.1f}  coseno={coseno:.2f}")
print("  -> sin normalizar, 'chunk largo' gana (11.0 > 4.0) aunque 'chunk corto' este mas alineado.")

# 2. Con los embeddings reales del corpus: confirmamos que normalizar no rompe nada.
with open(ESTADO / "02_chunks_con_embeddings.json", encoding="utf-8") as f:
    chunks = json.load(f)

vectores = np.array([chunk["embedding"] for chunk in chunks])
normas_antes = np.linalg.norm(vectores, axis=1)
normas_despues = np.linalg.norm(normalizar(vectores), axis=1)
print(f"\nNormas antes de normalizar: min={normas_antes.min():.3f} max={normas_antes.max():.3f}")
print(f"Normas despues de normalizar: min={normas_despues.min():.3f} max={normas_despues.max():.3f} (siempre 1.0)")
