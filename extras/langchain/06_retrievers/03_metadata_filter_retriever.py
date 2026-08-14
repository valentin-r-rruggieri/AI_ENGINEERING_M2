# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Filtros de metadata limitan qué documentos se pueden recuperar.
Son clave para tenant, idioma o permisos; no sustituyen autorización de aplicación.
# GUÍA DOCENTE
# CUÁNDO USAR: multi-tenant, idioma, área o documentos con permisos distintos.
# DIFERENCIA: buscar por similitud encuentra parecido; el filtro restringe el
# universo permitido antes de devolver texto.
# EN CLASE: mostrar que un filtro correcto es seguridad y no solo relevancia.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

store = FAISS.from_documents([Document(page_content="Plan público.", metadata={"tenant": "a"}), Document(page_content="Plan privado.", metadata={"tenant": "b"})], FakeEmbeddings(size=12))
documentos = store.similarity_search("plan", k=2, filter={"tenant": "a"})

for documento in documentos:
    print(documento.page_content, documento.metadata)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
