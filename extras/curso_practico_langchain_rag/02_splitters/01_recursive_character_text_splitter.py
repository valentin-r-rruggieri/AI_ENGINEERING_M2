"""RecursiveCharacterTextSplitter: opción inicial para prosa.
Prueba párrafos, líneas, palabras y caracteres para respetar chunk_size.
# GUÍA DOCENTE
# CUÁNDO USAR: primera opción para texto general, FAQs y documentación plana.
# DIFERENCIA: intenta párrafo, línea, espacio y carácter para respetar chunk_size;
# no entiende tokens ni estructura semántica profunda.
# EN CLASE: comparar overlap 0 vs 50 y buscar información partida entre chunks.
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

curso = Path(__file__).resolve().parents[1]
documentos = TextLoader(str(curso / "data" / "faq_empresa_saas.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50).split_documents(documentos)

print("Chunks:", len(chunks))
print(chunks[0].page_content)
print(chunks[0].metadata)
