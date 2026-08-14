# Este archivo experimenta con similarity_top_k, que controla cuantos nodes se devuelven por
# consulta. Un K alto aumenta cobertura pero puede introducir ruido y consumir mas contexto. Al
# ejecutarlo se ve la diferencia de cantidad entre dos configuraciones sobre el mismo indice.

# sys habilita imports del curso.
import sys
# Path localiza su raiz.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# DOCUMENTS alimenta el indice del ejemplo.
from shared.dataset import DOCUMENTS
# build_mock_index crea un indice local sin proveedor externo.
from shared.utils import build_mock_index

index = build_mock_index(DOCUMENTS)
print("top_k=1:", len(index.as_retriever(similarity_top_k=1).retrieve("horario")))
print("top_k=3:", len(index.as_retriever(similarity_top_k=3).retrieve("horario")))

# Resumen final: Top-K debe medirse porque mas resultados no significa mejor contexto para un LLM.
