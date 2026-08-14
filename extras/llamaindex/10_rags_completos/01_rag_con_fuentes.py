# Este archivo muestra como inspeccionar source_nodes de una respuesta de LlamaIndex. Las fuentes
# son necesarias para comprobar que el RAG uso evidencia y para mostrar citas al usuario. Usa mocks
# para funcionar localmente; al ejecutarlo imprime los nodes usados para responder una pregunta.

# sys permite importar corpus y helpers del curso.
import sys
# Path ubica la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS es el conocimiento que se indexa.
from shared.dataset import DOCUMENTS
# build_mock_index crea QueryEngine y show_nodes imprime fuentes.
from shared.utils import build_mock_index, show_nodes

response = build_mock_index(DOCUMENTS).as_query_engine(similarity_top_k=2).query("Como recupero mi contrasena?")
print(response)
show_nodes(response.source_nodes)

# Resumen final: una respuesta RAG debe entregar sus fuentes para ser verificable y auditable.
