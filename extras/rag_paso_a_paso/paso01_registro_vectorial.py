"""Paso 1: convertir los documentos en chunks con su metadata.

Todavia no hay embeddings: primero decidimos que es un "chunk" y que metadata
le asociamos. Los pasos siguientes van sumando funcionalidad arriba de esto.
"""
import json

from comun import DEPARTAMENTO_POR_FUENTE, DOCUMENTOS, ESTADO

# Cada linea no vacia de cada documento es un chunk (igual que en extras/embeddings).
chunks = []
for doc in DOCUMENTOS:
    for i, linea in enumerate(doc["text"].splitlines()):
        if linea.strip():
            chunks.append(
                {
                    "chunk_id": f"{doc['id']}_{i}",
                    "text": linea.strip(),
                    "metadata": {
                        "source": doc["source"],
                        "department": DEPARTAMENTO_POR_FUENTE[doc["source"]],
                    },
                }
            )

with open(ESTADO / "01_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Chunks generados: {len(chunks)}")
for chunk in chunks:
    print(f"  {chunk['chunk_id']} ({chunk['metadata']['department']}): {chunk['text']}")
print(f"\nGuardado en {ESTADO / '01_chunks.json'}")
