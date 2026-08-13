"""similarity_search_with_score: entrega documentos y sus scores.
El significado del score depende del backend; se inspecciona antes de fijar umbrales.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Plan Pro incluye analítica.", "La contraseña se restablece por email."], FakeEmbeddings(size=16))
resultados = store.similarity_search_with_score("plan", k=2)

for documento, score in resultados:
    print("Score:", round(float(score), 3), "|", documento.page_content)
