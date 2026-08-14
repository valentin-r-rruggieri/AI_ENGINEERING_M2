# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma: base local de producto, persistencia y respuesta grounded.
Usa data/rag_chromadb, un corpus distinto del RAG operativo de Pinecone.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG local persistente que sí responde con un LLM.
# FLUJO: corpus de producto -> Chroma en disco -> búsqueda -> prompt grounded.
# DIFERENCIA: Chroma se ejecuta localmente; Pinecone usa un índice cloud existente.
# EN CLASE: ejecutar dos veces y explicar persistencia, colección y fuentes.
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

curso = Path(__file__).resolve().parents[1]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_chroma import Chroma
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
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

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
