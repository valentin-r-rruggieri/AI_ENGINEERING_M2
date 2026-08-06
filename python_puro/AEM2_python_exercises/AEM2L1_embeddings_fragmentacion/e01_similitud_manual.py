"""E01: comparar coseno y producto punto sin API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cosine_similarity

# La consulta y los documentos comparten la misma dimensión vectorial.
consulta = [1, 2, 1]
documento_a = [2, 4, 2]
documento_b = [0, 1, 4]
# El coseno compara dirección: A es un múltiplo de la consulta y debe dar 1.0.
print("Coseno A:", round(cosine_similarity(consulta, documento_a), 3))
print("Coseno B:", round(cosine_similarity(consulta, documento_b), 3))
# El producto punto también depende de la longitud de los vectores.
print("Producto punto A:", sum(a*b for a, b in zip(consulta, documento_a)))
