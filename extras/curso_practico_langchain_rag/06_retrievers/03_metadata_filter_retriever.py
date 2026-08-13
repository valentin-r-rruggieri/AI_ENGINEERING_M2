"""Filtros de metadata limitan qué documentos se pueden recuperar.
Son clave para tenant, idioma o permisos; no sustituyen autorización de aplicación.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_documents([Document(page_content="Plan público.", metadata={"tenant": "a"}), Document(page_content="Plan privado.", metadata={"tenant": "b"})], FakeEmbeddings(size=12))
documentos = store.similarity_search("plan", k=2, filter={"tenant": "a"})

for documento in documentos:
    print(documento.page_content, documento.metadata)
