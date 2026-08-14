# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""create_retriever_tool hace que un agente pueda pedir evidencia documental.
La tool devuelve contexto; el agente decide si debe llamarla.
# GUÍA DOCENTE
# CUÁNDO USAR: agente que decide cuándo necesita consultar documentación.
# DIFERENCIA: RAG lineal recupera siempre; como tool el agente elige recuperar,
# lo que da flexibilidad pero también agrega costo y variabilidad.
# EN CLASE: comparar agente RAG con cadena RAG lineal.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.tools import create_retriever_tool

store = FAISS.from_texts(["La licencia Pro admite 10 usuarios.", "El soporte atiende por correo."], FakeEmbeddings(size=16))
herramienta = create_retriever_tool(store.as_retriever(search_kwargs={"k": 1}), "buscar_faq", "Busca evidencia en la FAQ.")

print(herramienta.invoke({"query": "¿Cuántos usuarios permite Pro?"}))

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
