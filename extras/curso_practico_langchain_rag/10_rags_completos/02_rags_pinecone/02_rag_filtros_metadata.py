"""RAG Pinecone filtrado por metadata.
El filtro se aplica antes de devolver chunks y sirve para tipo de documento, región o permisos.
# GUÍA DOCENTE
# CASO: buscar solo procedimientos clasificados como incidente.
# CUÁNDO USAR: recuperación con tipo, región, producto o permisos conocidos.
# DIFERENCIA: namespace separa espacios grandes; filter restringe documentos
# concretos dentro del namespace elegido.
# EN CLASE: explicar que metadata debe cargarse junto a cada vector.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=os.getenv("PINECONE_NAMESPACE", "operaciones"))
        recuperados = store.similarity_search("¿Cómo se documenta un incidente?", k=3, filter={"tipo": {"$eq": "incidente"}})
        for documento in recuperados:
            print(documento.metadata)
            print(documento.page_content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
