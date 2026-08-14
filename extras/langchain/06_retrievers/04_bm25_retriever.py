# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""BM25Retriever recupera por términos exactos, sin embeddings.
Es útil para códigos, nombres e identificadores; se combina con vectores en búsqueda híbrida.
# GUÍA DOCENTE
# CUÁNDO USAR: códigos de error, IDs, siglas y palabras exactas.
# DIFERENCIA: BM25 usa términos; embeddings usan significado. Una búsqueda
# híbrida combina ambas señales cuando el dominio mezcla texto y códigos.
# EN CLASE: buscar E401 y comparar con una pregunta semántica de soporte.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document

try:
    # Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
    from langchain_community.retrievers import BM25Retriever
    retriever = BM25Retriever.from_documents([Document(page_content="Error E401 de acceso."), Document(page_content="Soporte de facturación.")])
    print(retriever.invoke("E401")[0].page_content)
except ImportError:
    print("Instala BM25: pip install rank-bm25")

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
