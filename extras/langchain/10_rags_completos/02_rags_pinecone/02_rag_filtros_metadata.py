# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Pinecone filtrado por metadata.
El filtro se aplica antes de devolver chunks y sirve para tipo de documento, región o permisos.
# GUÍA DOCENTE
# CASO: buscar solo procedimientos clasificados como incidente.
# CUÁNDO USAR: recuperación con tipo, región, producto o permisos conocidos.
# DIFERENCIA: namespace separa espacios grandes; filter restringe documentos
# concretos dentro del namespace elegido.
# EN CLASE: explicar que metadata debe cargarse junto a cada vector.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from pinecone import Pinecone
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_openai import OpenAIEmbeddings
        # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
        from langchain_pinecone import PineconeVectorStore
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=os.getenv("PINECONE_NAMESPACE", "operaciones"))
        recuperados = store.similarity_search("¿Cómo se documenta un incidente?", k=3, filter={"tipo": {"$eq": "incidente"}})
        for documento in recuperados:
            print(documento.metadata)
            print(documento.page_content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
