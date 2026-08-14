# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""FakeEmbeddings: embedding local para probar flujos sin una API.
Sirve para pruebas técnicas; no representa similitud semántica real.
# GUÍA DOCENTE
# CUÁNDO USAR: pruebas locales de pipeline sin claves, costo ni red.
# DIFERENCIA: FakeEmbeddings solo mantiene dimensión e interfaz; OpenAIEmbeddings
# captura significado y es el que sirve para calidad real de retrieval.
# EN CLASE: usarlo para probar el flujo, nunca para evaluar precisión semántica.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=8)
vector = embeddings.embed_query("restablecer contraseña")

print("Dimensión:", len(vector))
print("Vector:", vector)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
