# Este archivo explica Settings, la configuracion global que LlamaIndex consulta al crear indices
# y query engines. Usa mocks para ejecutar sin proveedores externos y ver la estructura local.
# Al finalizar se confirma que embedding y LLM de prueba quedaron configurados.

# sys permite importar los helpers compartidos desde este script.
import sys
# Path calcula la raiz del curso para importar shared.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# setup_mock configura MockEmbedding y MockLLM en Settings.
from shared.utils import setup_mock

setup_mock()
print("Settings configurado con MockEmbedding y MockLLM.")

# Resumen final: Settings evita pasar el mismo proveedor a cada objeto de LlamaIndex.
