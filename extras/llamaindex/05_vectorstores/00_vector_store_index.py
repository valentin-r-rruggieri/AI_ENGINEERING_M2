# Este archivo crea VectorStoreIndex, el indice basico de LlamaIndex para asociar nodes con
# embeddings. Usa mocks para que se ejecute localmente y muestra que el indice ya puede exponer
# un retriever. Es la base sobre la que se construyen QueryEngine y RAG completos.

# sys permite importar el corpus y helpers del curso.
import sys
# Path ubica la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS contiene la evidencia que se indexara.
from shared.dataset import DOCUMENTS
# build_mock_index crea VectorStoreIndex con MockEmbedding.
from shared.utils import build_mock_index

index = build_mock_index(DOCUMENTS)
print("Indice creado:", type(index).__name__)

# Resumen final: VectorStoreIndex es el punto donde nodes y embeddings se convierten en retrieval.
