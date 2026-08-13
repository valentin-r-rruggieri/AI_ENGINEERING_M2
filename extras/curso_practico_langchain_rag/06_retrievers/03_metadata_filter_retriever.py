"""Filtros de metadata limitan qué documentos se pueden recuperar.
Son clave para tenant, idioma o permisos; no sustituyen autorización de aplicación.
# GUÍA DOCENTE
# CUÁNDO USAR: multi-tenant, idioma, área o documentos con permisos distintos.
# DIFERENCIA: buscar por similitud encuentra parecido; el filtro restringe el
# universo permitido antes de devolver texto.
# EN CLASE: mostrar que un filtro correcto es seguridad y no solo relevancia.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_documents([Document(page_content="Plan público.", metadata={"tenant": "a"}), Document(page_content="Plan privado.", metadata={"tenant": "b"})], FakeEmbeddings(size=12))
documentos = store.similarity_search("plan", k=2, filter={"tenant": "a"})

for documento in documentos:
    print(documento.page_content, documento.metadata)
