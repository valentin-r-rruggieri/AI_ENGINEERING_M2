# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Debugging RAG: inspecciona cada artefacto antes de cambiar el siguiente paso.
Document, chunks, retrieval, prompt y respuesta son evidencias separadas.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando la respuesta final parece incorrecta.
# DIFERENCIA: el error puede estar en carga, chunking, retrieval, prompt o LLM;
# cambiar el prompt no corrige un documento que nunca fue recuperado.
# EN CLASE: inspeccionar cada artefacto antes de modificar la siguiente etapa.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_text_splitters import RecursiveCharacterTextSplitter

documento = Document(page_content="El soporte se contacta por email. La facturación es mensual.", metadata={"source": "faq.txt"})
chunks = RecursiveCharacterTextSplitter(chunk_size=35, chunk_overlap=5).split_documents([documento])

print("Documento:", documento.metadata)
print("Chunks:", [chunk.page_content for chunk in chunks])

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
