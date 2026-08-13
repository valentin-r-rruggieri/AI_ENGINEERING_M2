"""as_retriever: adapta un vectorstore a la interfaz invoke(pregunta).
similarity devuelve los k vectores más cercanos a la pregunta.
# GUÍA DOCENTE
# CUÁNDO USAR: caso base de RAG, cuando bastan los vecinos más cercanos.
# DIFERENCIA: el vector store almacena/busca; el retriever expone invoke(query)
# y encapsula la estrategia para cadenas y agentes.
# EN CLASE: variar k y evaluar cuánta evidencia útil versus ruido llega al prompt.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Soporte por correo.", "Plan Pro incluye API.", "Facturación mensual."], FakeEmbeddings(size=24))
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
documentos = retriever.invoke("planes")

for documento in documentos:
    print(documento.page_content)
