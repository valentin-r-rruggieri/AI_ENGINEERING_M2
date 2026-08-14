# Este archivo convierte un QueryEngine en QueryEngineTool. Asi un agente puede elegir consultar
# el RAG como una herramienta entre varias opciones. Usa mocks y no requiere LLM real; el objetivo
# es ver como se describe una fuente de conocimiento para que un agente la pueda seleccionar.

# sys permite importar corpus y helpers.
import sys
# Path ubica la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# QueryEngineTool envuelve un query engine como herramienta.
from llama_index.core.tools import QueryEngineTool
# DOCUMENTS alimenta el indice de ejemplo.
from shared.dataset import DOCUMENTS
# build_mock_index crea el QueryEngine con mocks.
from shared.utils import build_mock_index

tool = QueryEngineTool.from_defaults(query_engine=build_mock_index(DOCUMENTS).as_query_engine(), name="politicas_empresa", description="Consulta politicas internas.")
print(tool.metadata.name, "->", tool.metadata.description)

# Resumen final: un agente necesita descripciones claras para elegir la herramienta RAG correcta.
