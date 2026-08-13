"""RAG Pinecone con regla de no inventar.
Incluso si retrieval devuelve chunks, el prompt exige responder sin evidencia cuando no responden la pregunta.
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
        recuperados = store.similarity_search("¿Cuál es el precio comercial del plan Enterprise?", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Responde solamente si el contexto tiene la respuesta. Si no, di exactamente: No tengo evidencia en los documentos.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuál es el precio comercial del plan Enterprise?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
