# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""as_retriever: adapta un vectorstore a la interfaz invoke(pregunta).
similarity devuelve los k vectores más cercanos a la pregunta.
# GUÍA DOCENTE
# CUÁNDO USAR: caso base de RAG, cuando bastan los vecinos más cercanos.
# DIFERENCIA: el vector store almacena/busca; el retriever expone invoke(query)
# y encapsula la estrategia para cadenas y agentes.
# EN CLASE: variar k y evaluar cuánta evidencia útil versus ruido llega al prompt.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Soporte por correo.", "Plan Pro incluye API.", "Facturación mensual."], FakeEmbeddings(size=24))
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
documentos = retriever.invoke("planes")

for documento in documentos:
    print(documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
