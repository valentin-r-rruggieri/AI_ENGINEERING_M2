# Este archivo presenta StorageContext, el mecanismo de LlamaIndex para guardar un indice y evitar
# reindexar en cada arranque. Usa mocks para no requerir modelos reales. Al ejecutarlo persiste el
# indice dentro de storage y muestra la ruta que luego se puede volver a cargar.

# sys permite importar el corpus comun.
import sys
# Path crea una ruta segura para el almacenamiento local.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS aporta el contenido indexado.
from shared.dataset import DOCUMENTS
# build_mock_index crea un indice local listo para persistir.
from shared.utils import build_mock_index

directory = Path(__file__).resolve().parents[1] / "storage" / "mock_index"
index = build_mock_index(DOCUMENTS)
index.storage_context.persist(persist_dir=str(directory))
print("Indice persistido en:", directory)

# Resumen final: persistir un indice ahorra tiempo, pero requiere versionar corpus y embeddings.
