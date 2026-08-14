# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""RAG con fuentes: cada chunk conserva origen e identificador.
Las citas hacen que la respuesta sea comprobable por quien la recibe.
# GUÍA DOCENTE
# CUÁNDO USAR: respuestas que deben poder auditarse o citar documentación.
# DIFERENCIA: una respuesta sin fuente puede parecer correcta sin ser verificable;
# la fuente conecta cada afirmación con un chunk recuperado.
# EN CLASE: pedir que cada respuesta tenga una cita y revisar si realmente la apoya.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import PromptTemplate

recuperados = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq_empresa_saas.txt", "chunk_id": "soporte-01"})]
formato = PromptTemplate.from_template("[{source} | {chunk_id}]\n{page_content}")
contexto = "\n".join(formato.format(page_content=documento.page_content, **documento.metadata) for documento in recuperados)

print(contexto)
print("Respuesta: Contacta soporte por email [faq_empresa_saas.txt | soporte-01].")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
