# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma con fuentes visibles.
Cada fragmento recuperado conserva metadata de origen para poder citarlo.
# GUÍA DOCENTE
# CASO: mesa de ayuda donde se necesita mostrar de qué fragmento sale la respuesta.
# CUÁNDO USAR: soporte, compliance o capacitación con revisión humana.
# DIFERENCIA: recupera igual que FAQ básica, pero hace visible source y posición
# para permitir verificar la afirmación.
# EN CLASE: explicar que una cita útil debe apuntar a evidencia relevante.
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

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    try:
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_chroma import Chroma
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_openai import OpenAIEmbeddings
        documentos = TextLoader(str(curso / "data" / "rag_chromadb" / "guia_producto_local.txt"), encoding="utf-8").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40, add_start_index=True).split_documents(documentos)
        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        store = Chroma.from_documents(chunks, embeddings, collection_name="soporte_fuentes", persist_directory=str(curso / "storage" / "chroma_soporte_fuentes"))
        recuperados = store.similarity_search("¿Cuál es el horario de soporte?", k=2)
        for documento in recuperados:
            print("Fuente:", documento.metadata.get("source"))
            print("Posición:", documento.metadata.get("start_index"))
            print(documento.page_content)
            print("-----")
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
