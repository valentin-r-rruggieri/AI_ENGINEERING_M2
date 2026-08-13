"""RAG Chroma con regla explícita para preguntas no respondibles.
El modelo recibe la instrucción de declarar falta de evidencia en vez de inventar.
# GUÍA DOCENTE
# CASO: pregunta sobre precio futuro que la guía no contiene.
# CUÁNDO USAR: siempre, como regla de seguridad y calidad de RAG.
# DIFERENCIA: recuperar chunks no demuestra que respondan la pregunta; el prompt
# debe obligar a reconocer la ausencia de evidencia.
# EN CLASE: revisar la respuesta y confirmar que no rellene datos imaginados.
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
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(documentos)
        store = Chroma.from_documents(chunks, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="sin_evidencia_producto", persist_directory=str(curso / "storage" / "chroma_sin_evidencia"))
        recuperados = store.similarity_search("¿Cuál será el precio del plan Pro en 2035?", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Contesta solo si el contexto contiene la respuesta. Si no, responde exactamente: No tengo evidencia en los documentos.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuál será el precio del plan Pro en 2035?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")
