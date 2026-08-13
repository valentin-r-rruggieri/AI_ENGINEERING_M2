"""RAG Pinecone con namespace.
Un namespace separa colecciones lógicas dentro del mismo índice, por ejemplo operaciones y producto.
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
        namespace = os.getenv("PINECONE_OPERATIONS_NAMESPACE", "operaciones")
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=namespace)
        recuperados = store.similarity_search("¿Cómo se gestiona un incidente?", k=2)
        print("Namespace:", namespace)
        for documento in recuperados:
            print(documento.page_content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
