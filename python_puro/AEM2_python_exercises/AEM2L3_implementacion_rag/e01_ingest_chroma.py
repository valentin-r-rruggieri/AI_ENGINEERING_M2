"""E01: ingesta en Chroma con embeddings reales."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chromadb
from common import embed_openai, split_words

root=Path(__file__).parent
# La ingesta separa el documento en unidades que luego podrán citarse por id.
chunks=split_words((root/"data"/"faq.txt").read_text(encoding="utf-8"),45,10,"faq")
# Los embeddings se calculan una vez durante indexación, no para cada respuesta.
vectors=embed_openai([c.content for c in chunks])
client=chromadb.PersistentClient(path=str(root/"data"/"chroma_db"))
collection=client.get_or_create_collection("faq",metadata={"hnsw:space":"cosine"})
# Reindexar reemplaza la colección para evitar mezclar versiones del corpus.
if collection.count(): collection.delete(ids=collection.get()["ids"])
# Documento, vector y metadata deben ocupar la misma posición en cada lista.
collection.add(ids=[c.chunk_id for c in chunks],documents=[c.content for c in chunks],embeddings=vectors,metadatas=[{"source":c.source} for c in chunks])
print("Chunks indexados:",collection.count())
