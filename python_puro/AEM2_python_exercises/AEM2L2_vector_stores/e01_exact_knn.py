"""E01: baseline exacto con similitud coseno."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import deterministic_embedding, split_words, top_k

root = Path(__file__).parent
chunks = split_words((root/"data"/"corpus.txt").read_text(encoding="utf-8"), 25, 5, "corpus")
# Este embedding local es suficiente para estudiar el ranking, no para evaluar semántica real.
vectors = [deterministic_embedding(chunk.content) for chunk in chunks]
# Búsqueda exacta: se compara la consulta contra todos los vectores disponibles.
for item in top_k(deterministic_embedding("licencias y vacaciones"), chunks, vectors, 3):
    print(item["chunk"]["chunk_id"], round(item["score"], 3))
