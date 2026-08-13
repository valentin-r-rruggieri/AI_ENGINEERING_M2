"""PromptTemplate también formatea cada Document recuperado.
Incluir fuente y chunk_id permite luego citar la evidencia en la respuesta.
"""
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

documentos = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq.txt", "chunk_id": "soporte-01"})]
formato = PromptTemplate.from_template("[Fuente: {source} | Chunk: {chunk_id}]\n{page_content}")
contexto = "\n\n".join(formato.format(page_content=documento.page_content, **documento.metadata) for documento in documentos)

print(contexto)
