"""Paso 2: convertir cada chunk del paso 1 en un embedding.

Docs: https://platform.openai.com/docs/guides/embeddings
"""
import json

from comun import EMBEDDING_MODEL, ESTADO, client

with open(ESTADO / "01_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Le pedimos a OpenAI un embedding por chunk, en una sola llamada.
textos = [chunk["text"] for chunk in chunks]
respuesta = client.embeddings.create(model=EMBEDDING_MODEL, input=textos)

for chunk, item in zip(chunks, respuesta.data):
    chunk["embedding"] = item.embedding

with open(ESTADO / "02_chunks_con_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

print(f"Embeddings generados: {len(chunks)} (dimension {len(chunks[0]['embedding'])})")
print(f"Guardado en {ESTADO / '02_chunks_con_embeddings.json'}")
