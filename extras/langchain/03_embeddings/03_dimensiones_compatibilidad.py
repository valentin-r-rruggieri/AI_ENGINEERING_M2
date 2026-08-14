# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Dimensión de embeddings: cada modelo produce una cantidad fija de números.
Un índice solo acepta la dimensión para la que fue creado; cambiar modelo exige reindexar.
# GUÍA DOCENTE
# CUÁNDO USAR: al cambiar de modelo de embeddings o configurar un índice cloud.
# DIFERENCIA: dimensión es cantidad de números; similitud es la regla para
# compararlos. Un índice no puede mezclar dimensiones distintas.
# EN CLASE: explicar que cambiar modelo obliga a reindexar, no solo a consultar.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings

embedding_8 = FakeEmbeddings(size=8).embed_query("soporte")
embedding_16 = FakeEmbeddings(size=16).embed_query("soporte")

print("Modelo A:", len(embedding_8), "dimensiones")
print("Modelo B:", len(embedding_16), "dimensiones")
print("Compatibles:", len(embedding_8) == len(embedding_16))

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
