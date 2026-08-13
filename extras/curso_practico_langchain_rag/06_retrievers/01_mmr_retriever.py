"""MMR combina relevancia con diversidad.
fetch_k reúne candidatos; k conserva resultados; lambda_mult regula diversidad.
"""
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS

store = FAISS.from_texts(["Plan Pro mensual.", "Plan Pro anual.", "Soporte por email.", "Seguridad de cuenta."], FakeEmbeddings(size=20))
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4, "lambda_mult": 0.5})
documentos = retriever.invoke("plan")

for documento in documentos:
    print(documento.page_content)
