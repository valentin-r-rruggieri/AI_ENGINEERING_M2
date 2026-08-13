"""Un dataset de evaluación asocia preguntas con evidencia esperada.
Incluye preguntas sin evidencia: la respuesta correcta es reconocer el límite.
"""
from langchain_core.documents import Document

casos = [
    {"pregunta": "¿Cómo contacto soporte?", "evidencia": Document(page_content="Soporte por email.", metadata={"chunk_id": "soporte"}), "respondible": True},
    {"pregunta": "¿Cuál será el precio en 2035?", "evidencia": None, "respondible": False},
]

for caso in casos:
    print(caso["pregunta"], "| respondible:", caso["respondible"])
