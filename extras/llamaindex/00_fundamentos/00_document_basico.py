# Este archivo presenta Document, la unidad de datos de LlamaIndex. Al ejecutarlo se ve como un
# texto y su metadata viajan juntos desde el inicio del RAG. No necesita modelos ni API key.
# Es el primer paso antes de convertir documentos en nodes e indices recuperables.

# Document representa conocimiento textual y metadata dentro de LlamaIndex.
from llama_index.core import Document

document = Document(text="La empresa permite trabajo remoto hasta tres dias.", metadata={"source": "rrhh.md", "category": "rrhh"})
print(document.get_content())
print(document.metadata)

# Resumen final: Document mantiene evidencia y procedencia juntas para retrieval y citas.
