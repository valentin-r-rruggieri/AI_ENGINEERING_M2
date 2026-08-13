"""OpenAIEmbeddings: genera vectores semánticos reales.
Documentos y consultas deben usar el mismo modelo y la misma dimensión.
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
