"""Hit@K comprueba presencia de evidencia; MRR penaliza evidencia tardía.
Se calculan para el mismo conjunto de preguntas al comparar configuraciones.
# GUÍA DOCENTE
# CUÁNDO USAR: al comparar embeddings, chunking o estrategia de retrieval.
# DIFERENCIA: Hit@K pregunta si recuperó; MRR considera posición. Una respuesta
# bien escrita puede seguir siendo mala si recuperó evidencia equivocada.
# EN CLASE: usar la misma pregunta para dos rankings y medir ambos.
"""
from langchain_core.documents import Document

resultados = [Document(page_content="No relevante", metadata={"id": "c3"}), Document(page_content="Soporte por email", metadata={"id": "soporte"})]
ids = [documento.metadata["id"] for documento in resultados]
posicion = ids.index("soporte") + 1

print("Hit@2:", "soporte" in ids)
print("MRR:", 1 / posicion)
