# Este archivo convierte Documents en nodes con SentenceSplitter. En LlamaIndex los nodes son las
# unidades que se embeben y recuperan, por eso decidir su tamano afecta directamente al RAG. Al
# ejecutarlo se imprimen los nodes del corpus comun y su metadata heredada.

# sys permite importar el dataset del curso.
import sys
# Path encuentra la raiz del curso.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# SentenceSplitter divide texto priorizando limites de oraciones.
from llama_index.core.node_parser import SentenceSplitter
# DOCUMENTS aporta el corpus de entrada.
from shared.dataset import DOCUMENTS
# show_nodes imprime contenido, score y metadata de nodes.
from shared.utils import show_nodes

nodes = SentenceSplitter(chunk_size=80, chunk_overlap=15).get_nodes_from_documents(DOCUMENTS)
show_nodes(nodes)

# Resumen final: node parsing prepara el nivel de granularidad con que LlamaIndex recupera evidencia.
