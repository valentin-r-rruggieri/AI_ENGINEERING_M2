"""CacheBackedEmbeddings: guarda embeddings ya calculados.
Reduce costo y latencia al repetir el mismo texto con el mismo modelo.
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
