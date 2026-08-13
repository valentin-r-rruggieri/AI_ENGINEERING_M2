"""Pinecone: consulta un índice cloud existente.
El índice debe tener la misma dimensión y métrica que el embedding elegido.
# GUÍA DOCENTE
# CUÁNDO USAR: corpus cloud, colaboración o escalado gestionado.
# DIFERENCIA: el índice se crea fuera del script y debe tener dimensión/métrica
# compatibles. Este ejemplo solo consulta estadísticas: no crea ni borra recursos.
# EN CLASE: revisar namespace, dimensión y fuente de los vectores antes de buscar.
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
