# Este archivo configura MockLLM, un modelo de prueba que permite construir query engines sin
# descargar ni pagar un LLM real. Su respuesta no es util para produccion, pero confirma que las
# piezas de LlamaIndex estan conectadas. Al ejecutarlo deja configurado el entorno local.

# sys permite importar el helper del curso.
import sys
# Path localiza la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# setup_mock registra MockLLM dentro de Settings.
from shared.utils import setup_mock

setup_mock()
print("MockLLM listo para construir QueryEngine sin proveedor externo.")

# Resumen final: MockLLM sirve para aprender arquitectura; una respuesta real requiere un LLM real.
