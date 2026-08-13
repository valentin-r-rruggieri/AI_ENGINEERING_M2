"""InMemoryVectorStore: almacena embeddings solo en memoria.
Es útil para aprender y probar; al cerrar Python el índice desaparece.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(FakeEmbeddings(size=24))
store.add_documents([Document(page_content="Soporte por email.", metadata={"tema": "soporte"}), Document(page_content="Plan Pro incluye API.", metadata={"tema": "planes"})])
resultados = store.similarity_search("planes", k=2)

for documento in resultados:
    print(documento.page_content, documento.metadata)
