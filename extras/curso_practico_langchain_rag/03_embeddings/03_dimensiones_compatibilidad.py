"""Dimensión de embeddings: cada modelo produce una cantidad fija de números.
Un índice solo acepta la dimensión para la que fue creado; cambiar modelo exige reindexar.
"""
from langchain_core.embeddings import FakeEmbeddings

embedding_8 = FakeEmbeddings(size=8).embed_query("soporte")
embedding_16 = FakeEmbeddings(size=16).embed_query("soporte")

print("Modelo A:", len(embedding_8), "dimensiones")
print("Modelo B:", len(embedding_16), "dimensiones")
print("Compatibles:", len(embedding_8) == len(embedding_16))
