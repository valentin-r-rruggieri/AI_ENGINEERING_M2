"""Un dataset de evaluación asocia preguntas con evidencia esperada.
Incluye preguntas sin evidencia: la respuesta correcta es reconocer el límite.
# GUÍA DOCENTE
# CUÁNDO USAR: antes de afirmar que un RAG 'funciona bien'.
# DIFERENCIA: ejemplos manuales enseñan; un dataset permite comparar cambios.
# Debe incluir preguntas respondibles y no respondibles con evidencia esperada.
# EN CLASE: pedir al grupo que diseñe tres casos por área de negocio.
"""
from langchain_core.documents import Document

casos = [
    {"pregunta": "¿Cómo contacto soporte?", "evidencia": Document(page_content="Soporte por email.", metadata={"chunk_id": "soporte"}), "respondible": True},
    {"pregunta": "¿Cuál será el precio en 2035?", "evidencia": None, "respondible": False},
]

for caso in casos:
    print(caso["pregunta"], "| respondible:", caso["respondible"])
