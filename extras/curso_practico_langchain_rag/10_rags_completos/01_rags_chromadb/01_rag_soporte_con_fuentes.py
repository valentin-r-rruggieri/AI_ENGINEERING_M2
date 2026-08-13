"""RAG Chroma con fuentes visibles.
Cada fragmento recuperado conserva metadata de origen para poder citarlo.
# GUÍA DOCENTE
# CASO: mesa de ayuda donde se necesita mostrar de qué fragmento sale la respuesta.
# CUÁNDO USAR: soporte, compliance o capacitación con revisión humana.
# DIFERENCIA: recupera igual que FAQ básica, pero hace visible source y posición
# para permitir verificar la afirmación.
# EN CLASE: explicar que una cita útil debe apuntar a evidencia relevante.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        documentos = TextLoader(str(curso / "data" / "rag_chromadb" / "guia_producto_local.txt"), encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40, add_start_index=True).split_documents(documentos)
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        store = Chroma.from_documents(chunks, embeddings, collection_name="soporte_fuentes", persist_directory=str(curso / "storage" / "chroma_soporte_fuentes"))
        recuperados = store.similarity_search("¿Cuál es el horario de soporte?", k=2)
        for documento in recuperados:
            print("Fuente:", documento.metadata.get("source"))
            print("Posición:", documento.metadata.get("start_index"))
            print(documento.page_content)
            print("-----")
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
