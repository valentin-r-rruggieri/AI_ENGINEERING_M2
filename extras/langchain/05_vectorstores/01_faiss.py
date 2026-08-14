# Este archivo forma parte del recorrido practico de LangChain y RAG.
# Lee la teoria inicial, ejecuta el ejemplo y modifica un parametro por vez para observar el efecto.

"""FAISS: vectorstore local con persistencia.
save_local guarda índice y metadata; al reabrir se usa el mismo embedding.
# GUÍA DOCENTE
# CUÁNDO USAR: búsqueda vectorial local rápida y sin servidor.
# DIFERENCIA: FAISS es un índice local; Chroma agrega colecciones y mayor manejo
# documental; Pinecone es un servicio cloud.
# EN CLASE: guardar, reabrir y comprobar que el índice sobrevive al proceso.
"""
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from pathlib import Path
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.documents import Document
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_core.embeddings import FakeEmbeddings
# Importa las herramientas necesarias para aplicar el concepto de LangChain de este ejemplo.
from langchain_community.vectorstores import FAISS

curso = Path(__file__).resolve().parents[1]
carpeta = curso / "storage" / "faiss_simple"
embeddings = FakeEmbeddings(size=16)
store = FAISS.from_documents([Document(page_content="Facturación mensual."), Document(page_content="Soporte por email.")], embeddings)
store.save_local(str(carpeta))
reabierto = FAISS.load_local(str(carpeta), embeddings, allow_dangerous_deserialization=True)

print(reabierto.similarity_search("soporte", k=1)[0].page_content)

# Resumen final: este ejercicio aplica la API de LangChain explicada arriba sobre un caso pequeno.
# Cambia una variable por vez y observa el resultado antes de combinarla con el resto del RAG.
