"""Hit@K comprueba presencia de evidencia; MRR penaliza evidencia tardía.
Se calculan para el mismo conjunto de preguntas al comparar configuraciones.
"""
from langchain_core.documents import Document

resultados = [Document(page_content="No relevante", metadata={"id": "c3"}), Document(page_content="Soporte por email", metadata={"id": "soporte"})]
ids = [documento.metadata["id"] for documento in resultados]
posicion = ids.index("soporte") + 1

print("Hit@2:", "soporte" in ids)
print("MRR:", 1 / posicion)
