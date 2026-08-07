"""Paso 1: convertir los documentos en objetos Document de LangChain.

Usamos RecursiveCharacterTextSplitter para el chunking (la misma herramienta
que en extras/embeddings/langchain), y le agregamos metadata de departamento
para poder practicar filtros mas adelante.
Docs: https://python.langchain.com/docs/how_to/recursive_text_splitter/
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from comun import DEPARTAMENTO_POR_FUENTE, DOCUMENTOS, guardar_documentos

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documentos = splitter.create_documents(
    texts=[doc["text"] for doc in DOCUMENTOS],
    metadatas=[{"source": doc["source"]} for doc in DOCUMENTOS],
)

# El splitter no numera los chunks ni sabe de departamentos: se lo agregamos nosotros.
for i, documento in enumerate(documentos):
    documento.metadata["chunk_id"] = f"chunk_{i}"
    documento.metadata["department"] = DEPARTAMENTO_POR_FUENTE[documento.metadata["source"]]

guardar_documentos(documentos, "01_documentos.json")

print(f"Documentos generados: {len(documentos)}")
for documento in documentos:
    print(f"  {documento.metadata['chunk_id']} ({documento.metadata['department']}): {documento.page_content}")
