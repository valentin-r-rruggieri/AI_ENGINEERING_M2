# Este archivo representa el reranking en dos etapas. Primero recupera varios candidatos de forma
# rapida y despues los reordena con una senal adicional basada en terminos. Al ejecutarlo se ve
# el ranking antes y despues; en produccion se puede reemplazar por un cross-encoder.
# sys habilita los imports de shared desde este script independiente.
import sys
# Path localiza la carpeta raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS es el corpus sobre el que se recuperan los candidatos.
from shared.dataset import DOCUMENTS
# lexical_search da el ranking inicial, tokens extrae terminos y show_documents permite compararlo.
from shared.utils import lexical_search, show_documents, tokens


def rerank(query, docs):
    # Un reranker trabaja sobre pocos candidatos y agrega una senal mas costosa o precisa.
    key_terms = set(tokens(query))
    return sorted(docs, key=lambda doc: (len(key_terms & set(tokens(doc.page_content))), doc.metadata["score"]), reverse=True)


# La primera etapa favorece velocidad; esta segunda etapa solo reordena sus cuatro candidatos.
initial = lexical_search("vacaciones de empleados", DOCUMENTS, k=4)
print("Antes:")
show_documents(initial)
print("\nDespues:")
show_documents(rerank("vacaciones de empleados", initial))
print("En un RAG real, esta funcion se puede reemplazar por un cross-encoder.")

# Resumen final: recuperar muchos candidatos rapido y rerankear pocos mejora el orden final.
# El reranking se mide por calidad adicional frente a la latencia que agrega al pipeline.
