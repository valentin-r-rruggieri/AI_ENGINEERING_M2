# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""similarity_search_with_score: entrega documentos y sus scores.
El significado del score depende del backend; se inspecciona antes de fijar umbrales.
# GUÍA DOCENTE
# CUÁNDO USAR: al calibrar retrieval y revisar calidad de resultados.
# DIFERENCIA: similarity_search devuelve documentos; con_score también muestra
# distancia/similitud. El número no es una probabilidad universal.
# EN CLASE: revisar top-k antes de elegir un threshold.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Plan Pro incluye analítica.", "La contraseña se restablece por email."], FakeEmbeddings(size=16))
resultados = store.similarity_search_with_score("plan", k=2)

for documento, score in resultados:
    print("Score:", round(float(score), 3), "|", documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
