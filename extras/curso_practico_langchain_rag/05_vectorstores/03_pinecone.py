"""Pinecone: consulta un índice cloud existente.
El índice debe tener la misma dimensión y métrica que el embedding elegido.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
    print("Faltan PINECONE_API_KEY o PINECONE_INDEX_NAME en .env")
else:
    try:
        from pinecone import Pinecone
        pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indice = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
        print(indice.describe_index_stats())
    except ImportError:
        print("Instala Pinecone: pip install pinecone")
