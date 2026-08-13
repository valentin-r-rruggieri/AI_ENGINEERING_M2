"""CacheBackedEmbeddings: guarda embeddings ya calculados.
Reduce costo y latencia al repetir el mismo texto con el mismo modelo.
# GUÍA DOCENTE
# CUÁNDO USAR: desarrollo o reindexaciones donde se repite el mismo contenido.
# DIFERENCIA: cache guarda vectores ya calculados; vector store guarda vectores
# para recuperar documentos. La cache se invalida al cambiar modelo o versión.
# EN CLASE: ejecutar dos veces y explicar por qué la segunda evita trabajo repetido.
"""
from pathlib import Path
from langchain_core.embeddings import FakeEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

curso = Path(__file__).resolve().parents[1]
cache = LocalFileStore(str(curso / "storage" / "embedding_cache"))
embeddings = CacheBackedEmbeddings.from_bytes_store(FakeEmbeddings(size=12), cache, namespace="curso-v1")
vectores = embeddings.embed_documents(["Plan Pro", "Plan Pro", "Soporte"])

print("Vectores:", len(vectores))
print("Dimensión:", len(vectores[0]))
