# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""La evaluación compara chunks recuperados con evidencia esperada.
Hit@K comprueba presencia y MRR premia que la evidencia aparezca al principio.
# GUÍA DOCENTE
# CUÁNDO USAR: antes de modificar prompt o modelo para mejorar un RAG.
# DIFERENCIA: Hit@K mide si hay evidencia; MRR mide qué tan arriba aparece.
# Ninguna de las dos evalúa redacción: solo calidad de recuperación.
# EN CLASE: mover la evidencia de posición y recalcular MRR.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document

resultados = [Document(page_content="Chunk irrelevante", metadata={"id": "c1"}), Document(page_content="Evidencia de soporte", metadata={"id": "soporte"})]
ids = [documento.metadata["id"] for documento in resultados]
posicion = ids.index("soporte") + 1

print("Hit@2:", "soporte" in ids)
print("MRR:", 1 / posicion)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
