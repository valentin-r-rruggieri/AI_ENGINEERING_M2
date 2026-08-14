# Este archivo hace visible el nucleo de RAG antes de agregar un LLM: recibe una pregunta, busca
# documentos y arma un bloque de contexto con fuentes. Funciona sin instalar modelos locales y
# permite inspeccionar si la evidencia es correcta. Al ejecutarlo se imprime pregunta y contexto.
# sys permite usar los modulos compartidos desde una ejecucion individual.
import sys
# Path localiza la raiz del curso.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS contiene la evidencia que respondera este RAG sin LLM.
from shared.dataset import DOCUMENTS
# lexical_search ofrece una recuperacion local sin depender aun de modelos descargados.
from shared.utils import lexical_search

question = "Cuantos dias de vacaciones tengo?"

# El retrieval siempre ocurre antes de generar; aqui se imprime evidencia para verlo con claridad.
results = lexical_search(question, DOCUMENTS, k=2)
context = "\n".join(f"[{doc.metadata['id']}] {doc.page_content}" for doc in results)

print("Pregunta:", question)
print("\nEvidencia recuperada:\n", context)
print("\nEste es el RAG minimo: pregunta -> retrieval -> contexto. El LLM se conecta en el siguiente paso.")

# Resumen final: un RAG empieza por encontrar evidencia y armar contexto, no por generar texto.
# Inspeccionar esta salida facilita detectar problemas de retrieval antes de usar un modelo local.
