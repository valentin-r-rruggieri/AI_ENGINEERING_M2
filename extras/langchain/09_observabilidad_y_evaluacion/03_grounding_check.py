# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Grounding significa que la respuesta se apoya en contexto recuperado.
El control básico es mostrar respuesta y fuentes juntas para revisión humana.
# GUÍA DOCENTE
# CUÁNDO USAR: revisión de si una respuesta está respaldada por sus fuentes.
# DIFERENCIA: grounding evalúa fidelidad al contexto; relevancia evalúa si el
# contexto correcto fue recuperado. Son fallos distintos.
# EN CLASE: comparar una respuesta correcta sin cita y una inventada con cita.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import ChatPromptTemplate

documentos = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq_empresa_saas.txt"})]
contexto = "\n".join(documento.page_content for documento in documentos)
prompt = ChatPromptTemplate.from_template("Usa solamente este contexto:\n{context}\n\nPregunta: {question}")

print(prompt.format(context=contexto, question="¿Cómo contacto soporte?"))
print("Fuente:", documentos[0].metadata["source"])

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
