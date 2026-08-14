# Este archivo compara dos configuraciones de SentenceSplitter sobre el mismo corpus. Muestra que
# chunk_size controla cuanto contexto guarda cada node y chunk_overlap conserva ideas entre limites.
# Al ejecutarlo se comparan las cantidades de nodes producidas por ambas decisiones.

# sys habilita importar datos compartidos.
import sys
# Path localiza la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SentenceSplitter permite variar chunk_size y chunk_overlap.
from llama_index.core.node_parser import SentenceSplitter
# DOCUMENTS se usa como entrada para ambas configuraciones.
from shared.dataset import DOCUMENTS

small = SentenceSplitter(chunk_size=50, chunk_overlap=0).get_nodes_from_documents(DOCUMENTS)
wide = SentenceSplitter(chunk_size=100, chunk_overlap=20).get_nodes_from_documents(DOCUMENTS)
print("Chunks pequenos:", len(small), "Chunks con overlap:", len(wide))

# Resumen final: chunking es una decision de retrieval que equilibra especificidad y contexto.
