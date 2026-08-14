# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""Un experimento registra configuración y resultado juntos.
Cambiar chunk_size, embedding y k a la vez impide saber qué produjo una mejora.
# GUÍA DOCENTE
# CUÁNDO USAR: cuando se ajusta una configuración de RAG repetidamente.
# DIFERENCIA: un resultado aislado no permite aprender; registrar modelo,
# chunk_size, k y métrica vuelve comparables los experimentos.
# EN CLASE: cambiar una variable por vez y conservar el resto fijo.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document

experimento = Document(
    page_content="Hit@3: 0.80",
    metadata={"embedding": "text-embedding-3-small", "chunk_size": 400, "k": 3, "version": "v1"},
)

print(experimento.page_content)
print(experimento.metadata)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
