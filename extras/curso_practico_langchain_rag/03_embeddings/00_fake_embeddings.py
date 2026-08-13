"""FakeEmbeddings: embedding local para probar flujos sin una API.
Sirve para pruebas técnicas; no representa similitud semántica real.
# GUÍA DOCENTE
# CUÁNDO USAR: pruebas locales de pipeline sin claves, costo ni red.
# DIFERENCIA: FakeEmbeddings solo mantiene dimensión e interfaz; OpenAIEmbeddings
# captura significado y es el que sirve para calidad real de retrieval.
# EN CLASE: usarlo para probar el flujo, nunca para evaluar precisión semántica.
"""
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=8)
vector = embeddings.embed_query("restablecer contraseña")

print("Dimensión:", len(vector))
print("Vector:", vector)
