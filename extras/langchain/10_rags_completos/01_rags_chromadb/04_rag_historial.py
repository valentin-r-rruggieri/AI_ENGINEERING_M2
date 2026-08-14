# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma con historial de chat.
El historial aclara la consulta; los documentos recuperados siguen siendo la única evidencia.
# GUÍA DOCENTE
# CASO: chatbot de producto con preguntas encadenadas.
# CUÁNDO USAR: el usuario refiere una conversación previa sin repetir el sujeto.
# DIFERENCIA: historial da contexto conversacional; retrieval entrega evidencia.
# No indexar el historial como si fuera documentación oficial.
# EN CLASE: identificar qué información llega por history y cuál por Chroma.
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
from langchain_core.messages import HumanMessage, AIMessage
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(documentos)
        store = Chroma.from_documents(chunks, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="historial_producto", persist_directory=str(curso / "storage" / "chroma_historial"))
        recuperados = store.similarity_search("plan Pro usuarios", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_messages([("system", "Responde solo con el contexto recuperado: {context}"), MessagesPlaceholder("history"), ("human", "{question}")])
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "history": [HumanMessage("Hablamos del plan Pro."), AIMessage("El plan Pro incluye API.")], "question": "¿Y cuántos usuarios permite?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
