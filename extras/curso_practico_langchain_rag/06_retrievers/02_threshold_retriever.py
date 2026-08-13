"""Similarity threshold descarta coincidencias demasiado débiles.
El valor se calibra con preguntas reales: no existe un umbral universal.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Soporte por correo.", "Facturación mensual."], FakeEmbeddings(size=16))
retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2, "k": 2})
documentos = retriever.invoke("soporte")

print("Documentos recuperados:", len(documentos))
