# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""MMR combina relevancia con diversidad.
fetch_k reúne candidatos; k conserva resultados; lambda_mult regula diversidad.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando top-k devuelve chunks casi idénticos.
# DIFERENCIA: similarity maximiza cercanía; MMR combina cercanía y diversidad.
# fetch_k reúne candidatos, k devuelve resultados y lambda_mult ajusta balance.
# EN CLASE: comparar las dos listas para la misma pregunta.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Plan Pro mensual.", "Plan Pro anual.", "Soporte por email.", "Seguridad de cuenta."], FakeEmbeddings(size=20))
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.5})
documentos = retriever.invoke("plan")

for documento in documentos:
    print(documento.page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
