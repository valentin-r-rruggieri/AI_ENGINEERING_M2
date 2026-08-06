"""E03: generar embeddings reales con OpenAI."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import embed_openai, split_words, write_json

root = Path(__file__).parent
# Primero creamos exactamente los chunks que luego serán enviados al proveedor.
chunks = split_words((root/"data"/"policies.txt").read_text(encoding="utf-8"), 45, 10, "policies")
# Esta llamada requiere OPENAI_API_KEY; common.py muestra un error claro si falta.
vectors = embed_openai([chunk.content for chunk in chunks])
# Guardamos metadatos, no los vectores, para registrar el experimento sin exponer datos.
write_json(root/"data"/"expected"/"openai_embeddings_metadata.json", {
    "model": "AEM2_EMBEDDING_MODEL", "chunks": len(chunks), "dimension": len(vectors[0])
})
print(f"Embeddings: {len(vectors)} | dimensión: {len(vectors[0])}")
