"""FakeEmbeddings: embedding local para probar flujos sin una API.
Sirve para pruebas técnicas; no representa similitud semántica real.
"""
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=8)
vector = embeddings.embed_query("restablecer contraseña")

print("Dimensión:", len(vector))
print("Vector:", vector)
