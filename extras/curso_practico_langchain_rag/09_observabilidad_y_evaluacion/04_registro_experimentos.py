"""Un experimento registra configuración y resultado juntos.
Cambiar chunk_size, embedding y k a la vez impide saber qué produjo una mejora.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando se ajusta una configuración de RAG repetidamente.
# DIFERENCIA: un resultado aislado no permite aprender; registrar modelo,
# chunk_size, k y métrica vuelve comparables los experimentos.
# EN CLASE: cambiar una variable por vez y conservar el resto fijo.
"""
from langchain_core.documents import Document

experimento = Document(
    page_content="Hit@3: 0.80",
    metadata={"embedding": "text-embedding-3-small", "chunk_size": 400, "k": 3, "version": "v1"},
)

print(experimento.page_content)
print(experimento.metadata)
