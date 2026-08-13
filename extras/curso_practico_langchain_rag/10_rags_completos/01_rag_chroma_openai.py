"""RAG Chroma: base local de producto, persistencia y respuesta grounded.
Usa data/rag_chromadb, un corpus distinto del RAG operativo de Pinecone.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG local persistente que sí responde con un LLM.
# FLUJO: corpus de producto -> Chroma en disco -> búsqueda -> prompt grounded.
# DIFERENCIA: Chroma se ejecuta localmente; Pinecone usa un índice cloud existente.
# EN CLASE: ejecutar dos veces y explicar persistencia, colección y fuentes.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

curso = Path(__file__).resolve().parents[1]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        documentos = TextLoader(str(curso / "data" / "rag_chromadb" / "guia_producto_local.txt"), encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60).split_documents(documentos)
        store = Chroma.from_documents(chunks, embeddings, collection_name="faq_curso", persist_directory=str(curso / "storage" / "rag_chroma_simple"))
        recuperados = store.similarity_search("¿Cómo contacto soporte del producto local?", k=3)
        contexto = "\n\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_messages([("system", "Usa solo el contexto. Sin evidencia responde: No tengo evidencia en los documentos."), ("human", "Contexto:\n{context}\n\nPregunta: {question}")])
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cómo contacto soporte del producto local?"})
        print(respuesta.content)
        print([documento.metadata.get("source") for documento in recuperados])
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
