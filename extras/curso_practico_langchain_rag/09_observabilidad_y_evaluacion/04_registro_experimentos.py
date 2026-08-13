"""Un experimento registra configuración y resultado juntos.
Cambiar chunk_size, embedding y k a la vez impide saber qué produjo una mejora.
"""
from langchain_core.documents import Document

experimento = Document(
    page_content="Hit@3: 0.80",
    metadata={"embedding": "text-embedding-3-small", "chunk_size": 400, "k": 3, "version": "v1"},
)

print(experimento.page_content)
print(experimento.metadata)
