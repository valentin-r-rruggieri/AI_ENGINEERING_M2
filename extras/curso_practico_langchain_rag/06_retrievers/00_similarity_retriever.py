"""as_retriever: adapta un vectorstore a la interfaz invoke(pregunta).
similarity devuelve los k vectores más cercanos a la pregunta.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Soporte por correo.", "Plan Pro incluye API.", "Facturación mensual."], FakeEmbeddings(size=24))
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
documentos = retriever.invoke("planes")

for documento in documentos:
    print(documento.page_content)
