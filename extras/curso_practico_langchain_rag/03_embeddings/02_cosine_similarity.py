"""Embeddings con InMemoryVectorStore.
El vector store aplica similitud entre embedding de pregunta y documentos.
# GUÍA DOCENTE
# CUÁNDO USAR: para demostrar que un vector store encuentra cercanía vectorial.
# DIFERENCIA: el usuario pregunta texto, LangChain lo embeddea y compara vectores;
# no es una búsqueda literal como BM25.
# EN CLASE: cambiar la pregunta y discutir por qué FakeEmbeddings no da calidad real.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(FakeEmbeddings(size=12))
store.add_texts(["El plan Pro incluye API.", "El soporte se contacta por email."])
resultados = store.similarity_search("¿Cómo contacto soporte?", k=2)

for documento in resultados:
    print(documento.page_content)
