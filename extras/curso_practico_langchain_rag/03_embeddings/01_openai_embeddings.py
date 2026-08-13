"""OpenAIEmbeddings: genera vectores semánticos reales.
Documentos y consultas deben usar el mismo modelo y la misma dimensión.
# GUÍA DOCENTE
# CUÁNDO USAR: indexación y consultas semánticas reales con OpenAI.
# DIFERENCIA: embed_documents procesa corpus; embed_query procesa la pregunta.
# Ambos deben usar modelo y dimensión compatibles con el mismo vector store.
# EN CLASE: mostrar longitud del vector, no sus números como si fueran interpretables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("OPENAI_API_KEY"):
    print("Falta OPENAI_API_KEY en .env")
else:
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    vector = embeddings.embed_query("¿Cómo restablezco mi contraseña?")
    print("Dimensión:", len(vector))
    print("Primeros valores:", vector[:5])
