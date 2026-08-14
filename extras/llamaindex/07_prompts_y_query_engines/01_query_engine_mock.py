# Este archivo crea un QueryEngine, la pieza de LlamaIndex que combina retrieval y sintesis de
# respuesta. Usa MockLLM y MockEmbedding para mostrar la llamada query sin credenciales. La salida
# es artificial, pero confirma el recorrido pregunta -> nodes -> respuesta.

# sys permite acceder al dataset comun.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS contiene el conocimiento del indice.
from shared.dataset import DOCUMENTS
# build_mock_index configura mocks y devuelve el indice.
from shared.utils import build_mock_index

response = build_mock_index(DOCUMENTS).as_query_engine(similarity_top_k=2).query("Cuantos dias de vacaciones tengo?")
print(response)

# Resumen final: QueryEngine simplifica la consulta RAG, pero sus respuestas dependen de retrieval y LLM.
