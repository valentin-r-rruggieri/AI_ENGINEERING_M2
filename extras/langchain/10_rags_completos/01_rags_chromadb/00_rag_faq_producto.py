# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma para preguntas frecuentes de producto.
Pipeline: TextLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → Chroma → ChatOpenAI.
# GUÍA DOCENTE
# CASO: asistente interno para dudas de planes, facturación y soporte de producto.
# CUÁNDO USAR: corpus local pequeño/mediano que debe persistir en la computadora.
# DIFERENCIA: este ejemplo cubre pregunta factual; los siguientes agregan fuentes,
# filtros, diversidad, historial y control de evidencia.
# EN CLASE: seguir el pipeline desde archivo hasta respuesta final.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.document_loaders import TextLoader
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_chroma import Chroma
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
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

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
