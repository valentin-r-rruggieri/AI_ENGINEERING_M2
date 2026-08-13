"""RAG con fuentes: cada chunk conserva origen e identificador.
Las citas hacen que la respuesta sea comprobable por quien la recibe.
"""
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

recuperados = [Document(page_content="El soporte se contacta por email.", metadata={"source": "faq_empresa_saas.txt", "chunk_id": "soporte-01"})]
formato = PromptTemplate.from_template("[{source} | {chunk_id}]\n{page_content}")
contexto = "\n".join(formato.format(page_content=documento.page_content, **documento.metadata) for documento in recuperados)

print(contexto)
print("Respuesta: Contacta soporte por email [faq_empresa_saas.txt | soporte-01].")
