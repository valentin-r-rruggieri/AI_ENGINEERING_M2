# Este archivo mide cuanto tarda crear un VectorStoreIndex local. La medicion separada permite
# distinguir problemas de indexacion de problemas de generacion. Usa mocks para que la latencia
# observada sea local y repetible. Al ejecutarlo imprime milisegundos de una indexacion pequena.

# sys permite importar el corpus del curso.
import sys
# Path localiza la raiz del curso.
from pathlib import Path
# time mide tiempo de pared en milisegundos.
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS es la entrada que se indexa.
from shared.dataset import DOCUMENTS
# build_mock_index crea el indice sin red.
from shared.utils import build_mock_index

start = time.perf_counter()
build_mock_index(DOCUMENTS)
print(f"Indexacion: {(time.perf_counter() - start) * 1000:.2f} ms")

# Resumen final: medir una etapa aislada facilita saber donde optimizar un RAG real.
