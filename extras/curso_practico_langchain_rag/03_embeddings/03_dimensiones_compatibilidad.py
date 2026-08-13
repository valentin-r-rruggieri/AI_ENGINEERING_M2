"""Dimensión de embeddings: cada modelo produce una cantidad fija de números.
Un índice solo acepta la dimensión para la que fue creado; cambiar modelo exige reindexar.
# GUÍA DOCENTE
# CUÁNDO USAR: al cambiar de modelo de embeddings o configurar un índice cloud.
# DIFERENCIA: dimensión es cantidad de números; similitud es la regla para
# compararlos. Un índice no puede mezclar dimensiones distintas.
# EN CLASE: explicar que cambiar modelo obliga a reindexar, no solo a consultar.
"""
from langchain_core.embeddings import FakeEmbeddings

embedding_8 = FakeEmbeddings(size=8).embed_query("soporte")
embedding_16 = FakeEmbeddings(size=16).embed_query("soporte")

print("Modelo A:", len(embedding_8), "dimensiones")
print("Modelo B:", len(embedding_16), "dimensiones")
print("Compatibles:", len(embedding_8) == len(embedding_16))
