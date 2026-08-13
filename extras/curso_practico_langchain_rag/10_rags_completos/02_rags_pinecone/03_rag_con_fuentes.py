"""RAG Pinecone que presenta fuentes junto con la respuesta.
La metadata source y chunk_id debe cargarse al hacer upsert de los documentos.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_pinecone import PineconeVectorStore
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=os.getenv("PINECONE_NAMESPACE", "operaciones"))
        recuperados = store.similarity_search("¿Cuánto tiempo se observa el despliegue inicial?", k=2)
        contexto = "\n\n".join(f"[{documento.metadata.get('source', 'sin fuente')}]\n{documento.page_content}" for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Usa solo el contexto y cita la fuente entre corchetes.\n{context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuánto tiempo se observa el despliegue inicial?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
