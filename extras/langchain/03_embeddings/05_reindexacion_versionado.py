# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Reindexación: un vectorstore se recrea al cambiar embedding o contenido.
Metadata conserva la versión usada para identificar índices desactualizados.
# GUÍA DOCENTE
# CUÁNDO USAR: mantenimiento de un RAG que cambia documentos o modelos.
# DIFERENCIA: actualizar metadata no recalcula un vector; al cambiar el texto,
# modelo o dimensión se debe crear una indexación compatible.
# EN CLASE: comparar versionado de contenido con versionado de embeddings.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.vectorstores import InMemoryVectorStore

documento = Document(page_content="Política de soporte.", metadata={"content_version": "v2", "embedding_model": "fake-16"})
store = InMemoryVectorStore(FakeEmbeddings(size=16))
store.add_documents([documento])
resultado = store.similarity_search("soporte", k=1)[0]

print(resultado.page_content)
print(resultado.metadata)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
