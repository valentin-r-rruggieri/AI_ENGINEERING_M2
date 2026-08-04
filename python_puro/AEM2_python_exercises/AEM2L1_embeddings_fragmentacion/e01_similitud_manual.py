"""E01: comparar coseno y producto punto sin API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cosine_similarity

consulta = [1, 2, 1]
documento_a = [2, 4, 2]
documento_b = [0, 1, 4]
print("Coseno A:", round(cosine_similarity(consulta, documento_a), 3))
print("Coseno B:", round(cosine_similarity(consulta, documento_b), 3))
print("Producto punto A:", sum(a*b for a, b in zip(consulta, documento_a)))

