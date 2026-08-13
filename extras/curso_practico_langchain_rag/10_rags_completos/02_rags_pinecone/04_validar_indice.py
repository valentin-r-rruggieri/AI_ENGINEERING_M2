"""Validación de un índice Pinecone antes de consultar.
La dimensión configurada debe coincidir con la dimensión del embedding de consulta.
# GUÍA DOCENTE
# CUÁNDO USAR: antes de insertar o consultar vectores en un índice cloud.
# DIFERENCIA: dimensión es una restricción técnica; namespace es una separación
# lógica; métrica es la forma de comparar vectores. No son lo mismo.
# EN CLASE: cambiar el embedding y ver por qué puede dejar de ser compatible.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

curso = Path(__file__).resolve().parents[2]
load_dotenv(curso / ".env")
if not all(os.getenv(variable) for variable in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]):
    print("Faltan variables de OpenAI o Pinecone en .env")
else:
    try:
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings
        pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        descripcion = pinecone.describe_index(os.environ["PINECONE_INDEX_NAME"])
        vector = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")).embed_query("prueba de dimensión")
        print("Dimensión índice:", descripcion.dimension)
        print("Dimensión embedding:", len(vector))
        print("Compatibles:", descripcion.dimension == len(vector))
    except ImportError:
        print("Instala Pinecone: pip install pinecone langchain-pinecone")
