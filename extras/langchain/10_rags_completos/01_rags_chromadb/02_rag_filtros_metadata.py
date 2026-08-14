# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Chroma con filtros de metadata.
Los filtros limitan recuperación por categoría antes de construir el contexto.
# GUÍA DOCENTE
# CASO: portal que separa facturación, planes y soporte.
# CUÁNDO USAR: cuando la pregunta ya conoce un ámbito o existe una restricción.
# DIFERENCIA: vector search decide relevancia; metadata filter decide qué documentos
# son admisibles. Ambos se necesitan en sistemas multiárea.
# EN CLASE: quitar el filtro y mostrar que podrían aparecer temas no deseados.
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
            Document(page_content="La factura se descarga desde Configuración.", metadata={"tema": "facturacion"}),
            Document(page_content="El plan Pro permite 15 usuarios.", metadata={"tema": "planes"}),
            Document(page_content="Soporte atiende de lunes a viernes.", metadata={"tema": "soporte"}),
        ]
        store = Chroma.from_documents(documentos, OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), collection_name="filtros_producto", persist_directory=str(curso / "storage" / "chroma_filtros"))
        recuperados = store.similarity_search("¿Dónde descargo una factura?", k=3, filter={"tema": "facturacion"})
        for documento in recuperados:
            print(documento.metadata, "|", documento.page_content)
    except ImportError:
        print("Instala Chroma: pip install langchain-chroma chromadb")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
