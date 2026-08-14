# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Pinecone: consulta un índice cloud de operaciones.
El índice existente debe haberse cargado con data/rag_pinecone; no crea ni elimina recursos.
# GUÍA DOCENTE
# CUÁNDO USAR: RAG cloud que consulta un índice ya creado en clase.
# DIFERENCIA: no construye el índice ni sube datos; valida y consulta el recurso
# existente. Chroma puede crear su colección local durante el ejemplo.
# EN CLASE: verificar dimensión, namespace y que el corpus cargado sea el esperado.
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

curso = Path(__file__).resolve().parents[1]
load_dotenv(curso / ".env")
documentos = TextLoader(str(curso / "data" / "rag_pinecone" / "manual_operaciones_cloud.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60).split_documents(documentos)
print("Corpus esperado en Pinecone:", len(chunks), "chunks de operaciones cloud.")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from pinecone import Pinecone
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_pinecone import PineconeVectorStore
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_core.prompts import ChatPromptTemplate
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indice = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=embeddings, namespace=os.getenv("PINECONE_NAMESPACE", ""))
        recuperados = store.similarity_search("¿Qué se debe hacer ante una alerta de latencia?", k=3)
        contexto = "\n\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Usa solo el contexto. Sin evidencia di que no la tienes.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Qué se debe hacer ante una alerta de latencia?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
