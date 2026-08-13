"""Chroma: colección vectorial local persistente.
Guarda documentos, embeddings y metadata en persist_directory.
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

try:
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
