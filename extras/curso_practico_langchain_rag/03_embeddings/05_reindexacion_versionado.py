"""Reindexación: un vectorstore se recrea al cambiar embedding o contenido.
Metadata conserva la versión usada para identificar índices desactualizados.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

documento = Document(page_content="Política de soporte.", metadata={"content_version": "v2", "embedding_model": "fake-16"})
store = InMemoryVectorStore(FakeEmbeddings(size=16))
store.add_documents([documento])
resultado = store.similarity_search("soporte", k=1)[0]

print(resultado.page_content)
print(resultado.metadata)
