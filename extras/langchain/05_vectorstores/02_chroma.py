# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Chroma: colección vectorial local persistente.
Guarda documentos, embeddings y metadata en persist_directory.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG local persistente con documentos y metadata.
# DIFERENCIA: InMemory es efímero; Chroma persiste una colección en disco;
# Pinecone delega la infraestructura en cloud.
# EN CLASE: explicar collection_name y persist_directory antes de ejecutar.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings

try:
    # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
    from langchain_chroma import Chroma
    curso = Path(__file__).resolve().parents[1]
    store = Chroma.from_documents(
        [Document(page_content="Las licencias se administran desde Configuración.", metadata={"tema": "licencias"})],
        FakeEmbeddings(size=16),
        collection_name="curso_demo",
        persist_directory=str(curso / "storage" / "chroma_simple"),
    )
    print(store.similarity_search("licencias", k=1)[0].page_content)
except ImportError:
    print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
