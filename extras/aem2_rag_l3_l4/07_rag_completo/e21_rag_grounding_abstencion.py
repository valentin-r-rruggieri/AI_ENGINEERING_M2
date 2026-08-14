# Este archivo muestra una regla fundamental de seguridad: sin evidencia recuperada no se debe
# inventar una respuesta. Ejecuta una pregunta cubierta y otra fuera del corpus para comparar
# grounding con abstencion. No requiere LLM ni credenciales porque la decision ocurre antes.
# sys permite importar los datos compartidos al ejecutar este ejercicio directamente.
import sys
# Path resuelve la raiz del curso desde el archivo actual.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# DOCUMENTS contiene la unica evidencia permitida para responder.
from shared.dataset import DOCUMENTS
# lexical_search devuelve el mejor Document y un score didactico de coincidencia.
from shared.utils import lexical_search


def answer_from_evidence(question: str) -> str:
    # La respuesta se habilita solo si la etapa de retrieval encontro alguna palabra en comun.
    best = lexical_search(question, DOCUMENTS, k=1)[0]
    if best.metadata["score"] <= 0:
        return "No tengo evidencia suficiente en las politicas disponibles."
    return f"Segun [{best.metadata['id']}]: {best.page_content}"


# La segunda pregunta esta fuera del corpus y muestra la abstencion en lugar de una invencion.
for question in ["Cuantos dias de vacaciones tengo?", "Cual es el menu del comedor?"]:
    print(f"P: {question}\nR: {answer_from_evidence(question)}\n")

print("La abstencion se basa en evidencia recuperada, no en la seguridad aparente de una respuesta.")

# Resumen final: la calidad de un RAG incluye reconocer cuando el corpus no contiene una respuesta.
# La abstencion basada en retrieval protege contra alucinaciones y hace el sistema mas confiable.
