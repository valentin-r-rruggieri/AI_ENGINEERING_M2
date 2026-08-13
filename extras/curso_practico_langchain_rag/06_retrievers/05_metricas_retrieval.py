"""La evaluación compara chunks recuperados con evidencia esperada.
Hit@K comprueba presencia y MRR premia que la evidencia aparezca al principio.
"""
from langchain_core.documents import Document

resultados = [Document(page_content="Chunk irrelevante", metadata={"id": "c1"}), Document(page_content="Evidencia de soporte", metadata={"id": "soporte"})]
ids = [documento.metadata["id"] for documento in resultados]
posicion = ids.index("soporte") + 1

print("Hit@2:", "soporte" in ids)
print("MRR:", 1 / posicion)
