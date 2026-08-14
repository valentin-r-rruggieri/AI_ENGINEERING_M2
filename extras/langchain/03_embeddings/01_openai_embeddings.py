# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""OpenAIEmbeddings: genera vectores semánticos reales.
Documentos y consultas deben usar el mismo modelo y la misma dimensión.
# GUÍA DOCENTE
# CUÁNDO USAR: indexación y consultas semánticas reales con OpenAI.
# DIFERENCIA: embed_documents procesa corpus; embed_query procesa la pregunta.
# Ambos deben usar modelo y dimensión compatibles con el mismo vector store.
# EN CLASE: mostrar longitud del vector, no sus números como si fueran interpretables.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
import os
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from dotenv import load_dotenv
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    vector = embeddings.embed_query("¿Cómo restablezco mi contraseña?")
    print("Dimensión:", len(vector))
    print("Primeros valores:", vector[:5])

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
