# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""InMemoryVectorStore: almacena embeddings solo en memoria.
Es útil para aprender y probar; al cerrar Python el índice desaparece.
# GUÍA DOCENTE
# CUÁNDO USAR: prototipos pequeños, pruebas y explicación visual del flujo.
# DIFERENCIA: InMemory desaparece al cerrar Python; FAISS o Chroma pueden
# persistir. No es adecuado para grandes volúmenes.
# EN CLASE: agregar documentos y recuperar; después reiniciar y notar que se pierde.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.vectorstores import InMemoryVectorStore

store = InMemoryVectorStore(FakeEmbeddings(size=24))
store.add_documents([Document(page_content="Soporte por email.", metadata={"tema": "soporte"}), Document(page_content="Plan Pro incluye API.", metadata={"tema": "planes"})])
resultados = store.similarity_search("planes", k=2)

for documento in resultados:
    print(documento.page_content, documento.metadata)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
