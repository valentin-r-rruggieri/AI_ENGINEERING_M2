# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""PromptTemplate también formatea cada Document recuperado.
Incluir fuente y chunk_id permite luego citar la evidencia en la respuesta.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando se necesita trazabilidad y citas en una respuesta RAG.
# DIFERENCIA: el chunk aporta contenido; el formatter conserva source/chunk_id
# para que la persona pueda comprobar la evidencia.
# EN CLASE: explicar por qué las fuentes van dentro del contexto y en la salida.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.prompts import PromptTemplate

documentos = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq.txt", "chunk_id": "soporte-01"})]
formato = PromptTemplate.from_template("[Fuente: {source} | Chunk: {chunk_id}]\n{page_content}")
contexto = "\n\n".join(formato.format(page_content=documento.page_content, **documento.metadata) for documento in documentos)

print(contexto)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
