# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma con MMR.
MMR reduce fragmentos repetidos: fetch_k busca candidatos y k conserva resultados diversos.
# GUÍA DOCENTE
# CASO: consulta amplia sobre un plan que tiene información distribuida.
# CUÁNDO USAR: cuando los primeros resultados son redundantes y falta cobertura.
# DIFERENCIA: similarity prioriza solo cercanía; MMR agrega diversidad entre chunks.
# EN CLASE: variar lambda_mult y describir el balance relevancia/diversidad.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document

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
        documentos = [
            Document(page_content="El plan Pro incluye API."),
            Document(page_content="El plan Pro permite 15 usuarios."),
            Document(page_content="El plan Starter permite 3 usuarios."),
            Document(page_content="Las facturas son mensuales."),
        ]
        store = Chroma.from_documents(documentos, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="planes_mmr", persist_directory=str(curso / "storage" / "chroma_mmr"))
        retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.5})
        recuperados = retriever.invoke("¿Qué incluye el plan Pro?")
        for documento in recuperados:
            print(documento.page_content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
