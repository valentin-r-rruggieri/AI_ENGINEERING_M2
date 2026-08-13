"""RAG Chroma para preguntas frecuentes de producto.
Pipeline: TextLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → Chroma → ChatOpenAI.
# GUÍA DOCENTE
# CASO: asistente interno para dudas de planes, facturación y soporte de producto.
# CUÁNDO USAR: corpus local pequeño/mediano que debe persistir en la computadora.
# DIFERENCIA: este ejemplo cubre pregunta factual; los siguientes agregan fuentes,
# filtros, diversidad, historial y control de evidencia.
# EN CLASE: seguir el pipeline desde archivo hasta respuesta final.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        documentos = TextLoader(str(curso / "data" / "rag_chromadb" / "guia_producto_local.txt"), encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50).split_documents(documentos)
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        store = Chroma.from_documents(chunks, embeddings, collection_name="faq_producto", persist_directory=str(curso / "storage" / "chroma_faq_producto"))
        recuperados = store.similarity_search("¿Cuántos usuarios permite el plan Pro?", k=2)
        contexto = "\n\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Usa solamente el contexto.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuántos usuarios permite el plan Pro?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
