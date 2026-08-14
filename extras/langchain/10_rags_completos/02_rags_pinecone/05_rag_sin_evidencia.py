# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG Pinecone con regla de no inventar.
Incluso si retrieval devuelve chunks, el prompt exige responder sin evidencia cuando no responden la pregunta.
# GUÍA DOCENTE
# CASO: se pregunta precio comercial a un índice que solo conoce operaciones.
# CUÁNDO USAR: todo RAG de dominio limitado debe contemplar esta salida.
# DIFERENCIA: Pinecone puede devolver similitudes aun cuando no haya respuesta;
# la política grounded decide no convertir esos chunks en una invención.
# EN CLASE: relacionar no-evidencia con threshold y evaluación humana.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
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
        indice = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
        store = PineconeVectorStore(index=indice, embedding=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")), namespace=os.getenv("PINECONE_NAMESPACE", "operaciones"))
        recuperados = store.similarity_search("¿Cuál es el precio comercial del plan Enterprise?", k=2)
        contexto = "\n".join(documento.page_content for documento in recuperados)
        prompt = ChatPromptTemplate.from_template("Responde solamente si el contexto tiene la respuesta. Si no, di exactamente: No tengo evidencia en los documentos.\nContexto: {context}\nPregunta: {question}")
        respuesta = (prompt | ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)).invoke({"context": contexto, "question": "¿Cuál es el precio comercial del plan Enterprise?"})
        print(respuesta.content)
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
