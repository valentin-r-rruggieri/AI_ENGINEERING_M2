"""FAISS: vectorstore local con persistencia.
save_local guarda índice y metadata; al reabrir se usa el mismo embedding.
# GUÍA DOCENTE
# CUÁNDO USAR: búsqueda vectorial local rápida y sin servidor.
# DIFERENCIA: FAISS es un índice local; Chroma agrega colecciones y mayor manejo
# documental; Pinecone es un servicio cloud.
# EN CLASE: guardar, reabrir y comprobar que el índice sobrevive al proceso.
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

curso = Path(__file__).resolve().parents[1]
carpeta = curso / "storage" / "faiss_simple"
embeddings = FakeEmbeddings(size=16)
store = FAISS.from_documents([Document(page_content="Facturación mensual."), Document(page_content="Soporte por email.")], embeddings)
store.save_local(str(carpeta))
reabierto = FAISS.load_local(str(carpeta), embeddings, allow_dangerous_deserialization=True)

print(reabierto.similarity_search("soporte", k=1)[0].page_content)
