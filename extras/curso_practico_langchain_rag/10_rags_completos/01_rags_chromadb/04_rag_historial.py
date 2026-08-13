"""RAG Chroma con historial de chat.
El historial aclara la consulta; los documentos recuperados siguen siendo la única evidencia.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        documentos = TextLoader(str(curso / "data" / "rag_chromadb" / "guia_producto_local.txt"), encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(documentos)
        store = Chroma.from_documents(chunks, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="historial_producto", persist_directory=str(curso / "storage" / "chroma_historial"))
        recuperados = store.similarity_search("plan Pro usuarios", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_messages([("system", "Responde solo con el contexto recuperado: {context}"), MessagesPlaceholder("history"), ("human", "{question}")])
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "history": [HumanMessage("Hablamos del plan Pro."), AIMessage("El plan Pro incluye API.")], "question": "¿Y cuántos usuarios permite?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
