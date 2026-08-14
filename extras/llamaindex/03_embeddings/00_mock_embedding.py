# Este archivo configura MockEmbedding para crear indices sin API key ni descarga de modelos. Los
# vectores no representan semantica real, pero permiten aprender el contrato de LlamaIndex y ver
# como un indice recibe un proveedor de embeddings mediante Settings.

# sys permite usar la configuracion local del curso.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# setup_mock registra MockEmbedding dentro de Settings.
from shared.utils import setup_mock

setup_mock()
print("MockEmbedding configurado para ejercicios locales.")

# Resumen final: los mocks validan el flujo tecnico antes de medir calidad semantica real.
