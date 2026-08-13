"""IDs estables permiten actualizar y eliminar documentos concretos.
Cada backend tiene operaciones propias; aquí InMemoryVectorStore ilustra add y delete.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(FakeEmbeddings(size=12))
ids = store.add_documents([Document(page_content="Política de soporte v1")], ids=["soporte-v1"])
print("ID creado:", ids)
store.delete(ids=["soporte-v1"])
print("Índice sin el documento:", store.similarity_search("soporte", k=1))
