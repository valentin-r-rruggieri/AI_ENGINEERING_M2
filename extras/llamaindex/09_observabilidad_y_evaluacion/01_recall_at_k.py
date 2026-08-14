# Este archivo calcula Recall@K sobre consultas etiquetadas. La metrica responde si el retriever
# llevo al menos un documento relevante al contexto, antes de evaluar la redaccion del LLM. Usa
# mocks para explicar el calculo; conecta embeddings reales para obtener una medicion significativa.

# sys permite cargar dataset y utilidades compartidas.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS y EVALUATION_QUERIES aportan evidencia y respuestas esperadas.
from shared.dataset import DOCUMENTS, EVALUATION_QUERIES
# build_mock_index crea el retriever de prueba.
from shared.utils import build_mock_index

retriever = build_mock_index(DOCUMENTS).as_retriever(similarity_top_k=3)
scores = []
for item in EVALUATION_QUERIES:
    ids = {node.node.metadata["id"] for node in retriever.retrieve(item["query"])}
    scores.append(float(bool(ids & item["relevant_ids"])))
print(f"Recall@3: {sum(scores) / len(scores):.2f}")

# Resumen final: Recall@K comprueba cobertura de evidencia y no calidad de redaccion.
