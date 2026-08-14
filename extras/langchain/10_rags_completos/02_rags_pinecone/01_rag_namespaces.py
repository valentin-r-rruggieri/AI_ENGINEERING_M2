# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Pinecone con namespace.
Un namespace separa colecciones lógicas dentro del mismo índice, por ejemplo operaciones y producto.
# GUÍA DOCENTE
# CASO: mismo índice para varios dominios, como producto y operaciones.
# CUÁNDO USAR: separar conjuntos lógicos sin crear un índice por cada uno.
# DIFERENCIA: namespace organiza/aisla vectores; metadata filtra atributos dentro
# de ese conjunto. Elegir namespace incorrecto devuelve contexto equivocado.
# EN CLASE: comparar nombres de namespace y discutir aislamiento de datos.
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
        namespace = os.getenv("PINECONE_OPERATIONS_NAMESPACE", "operaciones")
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=namespace)
        recuperados = store.similarity_search("¿Cómo se gestiona un incidente?", k=2)
        print("Namespace:", namespace)
        for documento in recuperados:
            print(documento.page_content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
