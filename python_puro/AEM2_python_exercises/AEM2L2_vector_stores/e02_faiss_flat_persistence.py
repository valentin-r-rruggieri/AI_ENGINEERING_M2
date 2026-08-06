"""E02: FAISS Flat persistente."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from common import deterministic_embedding, normalize, split_words, write_json

root = Path(__file__).parent
chunks = split_words((root/"data"/"corpus.txt").read_text(encoding="utf-8"), 25, 5, "corpus")
# Normalizar permite que producto interno represente la similitud coseno.
matrix = np.asarray([normalize(deterministic_embedding(c.content)) for c in chunks], dtype="float32")
# IndexFlatIP es el baseline exacto: no aproxima vecinos, pero escala linealmente.
index = faiss.IndexFlatIP(matrix.shape[1]); index.add(matrix)
# El índice guarda vectores; la metadata se persiste por separado para reconstruir chunks.
faiss.write_index(index, str(root/"data"/"index.faiss"))
write_json(root/"data"/"metadata.json", [c.to_dict() for c in chunks])
scores, ids = index.search(np.asarray([normalize(deterministic_embedding("integraciones"))], dtype="float32"), 3)
# Los ids devueltos son posiciones de la matriz y deben cruzarse con metadata.json.
print(list(zip(ids[0].tolist(), [round(float(v), 3) for v in scores[0]])))
