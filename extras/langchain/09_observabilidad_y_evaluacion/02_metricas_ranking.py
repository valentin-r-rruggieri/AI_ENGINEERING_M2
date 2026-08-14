# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Hit@K comprueba presencia de evidencia; MRR penaliza evidencia tardía.
Se calculan para el mismo conjunto de preguntas al comparar configuraciones.
# GUÍA DOCENTE
# CUÁNDO USAR: al comparar embeddings, chunking o estrategia de retrieval.
# DIFERENCIA: Hit@K pregunta si recuperó; MRR considera posición. Una respuesta
# bien escrita puede seguir siendo mala si recuperó evidencia equivocada.
# EN CLASE: usar la misma pregunta para dos rankings y medir ambos.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document

resultados = [Document(page_content="No relevante", metadata={"id": "c3"}), Document(page_content="Soporte por email", metadata={"id": "soporte"})]
ids = [documento.metadata["id"] for documento in resultados]
posicion = ids.index("soporte") + 1

print("Hit@2:", "soporte" in ids)
print("MRR:", 1 / posicion)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
