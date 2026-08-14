# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma con regla explícita para preguntas no respondibles.
El modelo recibe la instrucción de declarar falta de evidencia en vez de inventar.
# GUÍA DOCENTE
# CASO: pregunta sobre precio futuro que la guía no contiene.
# CUÁNDO USAR: siempre, como regla de seguridad y calidad de RAG.
# DIFERENCIA: recuperar chunks no demuestra que respondan la pregunta; el prompt
# debe obligar a reconocer la ausencia de evidencia.
# EN CLASE: revisar la respuesta y confirmar que no rellene datos imaginados.
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
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(documentos)
        store = Chroma.from_documents(chunks, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="sin_evidencia_producto", persist_directory=str(curso / "storage" / "chroma_sin_evidencia"))
        recuperados = store.similarity_search("¿Cuál será el precio del plan Pro en 2035?", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Contesta solo si el contexto contiene la respuesta. Si no, responde exactamente: No tengo evidencia en los documentos.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuál será el precio del plan Pro en 2035?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
