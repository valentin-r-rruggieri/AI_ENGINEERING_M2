# Este archivo usa el retriever que entrega VectorStoreIndex. El retriever recibe una pregunta y
# devuelve NodeWithScore, es decir evidencia con un puntaje de similitud. Usa mocks para mostrar
# la interfaz; la calidad real aparece al conectar embeddings semanticos reales.

# sys permite importar piezas compartidas.
import sys
# Path localiza la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS es la fuente del indice local.
from shared.dataset import DOCUMENTS
# build_mock_index crea el indice y show_nodes imprime resultados.
from shared.utils import build_mock_index, show_nodes

retriever = build_mock_index(DOCUMENTS).as_retriever(similarity_top_k=2)
show_nodes(retriever.retrieve("vacaciones"))

# Resumen final: Retriever separa la busqueda de evidencia de la generacion de una respuesta.
