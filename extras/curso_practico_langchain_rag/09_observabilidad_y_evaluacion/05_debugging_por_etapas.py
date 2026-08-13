"""Debugging RAG: inspecciona cada artefacto antes de cambiar el siguiente paso.
Document, chunks, retrieval, prompt y respuesta son evidencias separadas.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

documento = Document(page_content="El soporte se contacta por email. La facturación es mensual.", metadata={"source": "faq.txt"})
chunks = RecursiveCharacterTextSplitter(chunk_size=35, chunk_overlap=5).split_documents([documento])

print("Documento:", documento.metadata)
print("Chunks:", [chunk.page_content for chunk in chunks])
