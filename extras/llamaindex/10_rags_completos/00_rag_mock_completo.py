# Este archivo arma un RAG completo local con las piezas basicas de LlamaIndex: Documents, indice,
# retriever, QueryEngine y MockLLM. La respuesta es artificial porque no hay modelo real, pero el
# flujo se ejecuta de punta a punta y sirve para validar estructura antes de agregar proveedores.

# sys habilita imports desde el curso.
import sys
# Path localiza su raiz.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS contiene las politicas que respondera el RAG.
from shared.dataset import DOCUMENTS
# build_mock_index crea el indice con proveedores didacticos.
from shared.utils import build_mock_index

query_engine = build_mock_index(DOCUMENTS).as_query_engine(similarity_top_k=2)
print(query_engine.query("Puedo trabajar remoto?"))

# Resumen final: un RAG completo une indexacion, retrieval y sintesis bajo una sola consulta.
