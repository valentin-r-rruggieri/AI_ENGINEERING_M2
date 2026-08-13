"""RAG Chroma con filtros de metadata.
Los filtros limitan recuperación por categoría antes de construir el contexto.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        documentos = [
            Document(page_content="La factura se descarga desde Configuración.", metadata={"tema": "facturacion"}),
            Document(page_content="El plan Pro permite 15 usuarios.", metadata={"tema": "planes"}),
            Document(page_content="Soporte atiende de lunes a viernes.", metadata={"tema": "soporte"}),
        ]
        store = Chroma.from_documents(documentos, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="filtros_producto", persist_directory=str(curso / "storage" / "chroma_filtros"))
        recuperados = store.similarity_search("¿Dónde descargo una factura?", k=3, filter={"tema": "facturacion"})
        for documento in recuperados:
            print(documento.metadata, "|", documento.page_content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
